document.addEventListener("DOMContentLoaded", () => {
    const streakEl = document.getElementById("streak");
    const btn = document.getElementById("increment");
  
    function updateStreak() {
      chrome.storage.local.get("streak", (data) => {
        streakEl.textContent = data.streak || 0;
      });
    }
  
    btn.onclick = () => {
      chrome.runtime.sendMessage({ type: "incrementStreak" });
      updateStreak();
    };
  
    updateStreak();
  });
  