document.getElementById("analyzeBtn").addEventListener("click", analyzeFiles);

async function analyzeFiles() {
    const fileInput = document.getElementById("audioFiles");
    const tableBody = document.getElementById("resultsTable");
    const resultsSection = document.getElementById("resultsSection");

    const button = document.getElementById("analyzeBtn");
    const btnText = document.getElementById("btnText");
    const btnLoader = document.getElementById("btnLoader");

    if (fileInput.files.length === 0) {
        alert("Please select at least one audio file");
        return;
    }

    // 🔄 START LOADING STATE
    button.disabled = true;
    btnText.innerText = "Analyzing audio...";
    btnLoader.classList.remove("hidden");

    resultsSection.classList.add("hidden");
    tableBody.innerHTML = "";

    const formData = new FormData();
    for (let file of fileInput.files) {
        formData.append("files", file);
    }

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        data.results.forEach(item => {
            const emotionClass = `emotion-${item.emotion.toLowerCase()}`;

            const row = `
                <tr>
                    <td>${item.file_name}</td>
                    <td class="${emotionClass}">${item.emotion}</td>
                    <td>${item.confidence}%</td>
                </tr>
            `;
            tableBody.innerHTML += row;
        });

        resultsSection.classList.remove("hidden");

    } catch (error) {
        alert("Error analyzing audio");
        console.error(error);
    }

    // ✅ STOP LOADING STATE
    button.disabled = false;
    btnText.innerText = "Analyze Emotions";
    btnLoader.classList.add("hidden");
}
