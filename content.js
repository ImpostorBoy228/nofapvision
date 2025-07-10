(async () => {
  // Загрузка TensorFlow.js
  await import('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.21.0/dist/tf.min.js');

  // Создание видеоэлемента
  const video = document.createElement("video");
  video.style.position = "fixed";
  video.style.bottom = "10px";
  video.style.right = "10px";
  video.style.width = "160px";
  video.style.height = "120px";
  video.style.zIndex = 999999;
  video.style.border = "3px solid red";
  document.body.appendChild(video);

  try {
      // Получение доступа к веб-камере
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject = stream;
      video.play();

      // Загрузка модели
      const model = await tf.loadLayersModel(chrome.runtime.getURL('nofap_model.h5'));

      // Функция предобработки кадра
      function preprocessFrame(frame) {
          // Изменение размера до 32x32 и нормализация
          let tensor = tf.browser.fromPixels(frame)
              .resizeNearestNeighbor([32, 32])
              .toFloat()
              .div(tf.scalar(255.0));
          // Добавление batch dimension
          return tensor.expandDims(0);
      }

      // Захват и обработка кадров
      const canvas = document.createElement('canvas');
      canvas.width = 32;
      canvas.height = 32;
      const ctx = canvas.getContext('2d');

      setInterval(async () => {
          // Захват кадра
          ctx.drawImage(video, 0, 0, 32, 32);
          const frame = ctx.getImageData(0, 0, 32, 32);

          // Предобработка и предсказание
          const processedFrame = preprocessFrame(frame);
          const prediction = await model.predict(processedFrame).data();
          const isSuspicious = prediction[0] > 0.5; // Порог для бинарной классификации

          if (isSuspicious) {
              const userResponse = confirm("ТЫ ЧТО ДРОЧИЛ?!?!?! Да/Нет");
              if (userResponse) {
                  chrome.runtime.sendMessage({ type: "resetStreak" });
              }
          }

          // Очистка памяти
          tf.dispose([processedFrame]);
      }, 3000);

  } catch (err) {
      console.error("Ошибка: ", err);
  }
})();