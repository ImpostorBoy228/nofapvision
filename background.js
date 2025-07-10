chrome.runtime.onInstalled.addListener(() => {
    chrome.storage.local.set({ streak: 0 });
  });
  
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.type === "resetStreak") {
      chrome.storage.local.set({ streak: 0 });
      chrome.notifications.create({
        type: "basic",
        iconUrl: "icon48.png",
        title: "NoFapVision",
        message: "Стрик сброшен! Попробуй снова, Повелитель!",
      });
    }
    if (request.type === "incrementStreak") {
      chrome.storage.local.get("streak", (data) => {
        let newStreak = (data.streak || 0) + 1;
        chrome.storage.local.set({ streak: newStreak });
        chrome.notifications.create({
          type: "basic",
          iconUrl: "icon48.png",
          title: "NoFapVision",
          message: `Стрик: ${newStreak} дней! Keep going, Dark Lord!`,
        });
      });
    }
  });
  