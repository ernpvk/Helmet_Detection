function updateCounts() {
  fetch("/counts")
    .then((response) => response.json())
    .then((data) => {
      document.querySelector("#helmet").textContent = data.helmet;
      document.querySelector("#no_helmet").textContent = data.no_helmet;

      const notificationDiv = document.getElementById("notification");
      if (data.no_helmet > 0) {
        notificationDiv.textContent = `⚠️ ${data.no_helmet} person(s) detected without a helmet!`;
        notificationDiv.style.display = "block";
      } else {
        notificationDiv.style.display = "none";
      }
    })
    .catch((error) => console.error("Error fetching count data:", error));
}

// Update counts every second
setInterval(updateCounts, 1000);
