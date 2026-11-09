async () => {
    // Request permission directly from the explicit user click. Gradio's
    // webcam has separate permission and streaming controls; start both.
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
        return "Камераға қолжетім жоқ. Бетті кәдімгі Chrome немесе Edge терезесінде http://127.0.0.1:7860 мекенжайымен ашыңыз.";
    }
    const root = document.getElementById("browser-camera");
    // Pause the other tab's recognition before starting this webcam stream.
    // Both camera panels retain their preview and can be started again.
    for (const other of document.querySelectorAll("#browser-camera, #game-camera")) {
        if (other !== root) {
            other.querySelector('[title="stop recording"]')?.closest("button")?.click();
        }
    }
    const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
    const errorMessage = error => {
        const help = {
            NotAllowedError: "Камераға рұқсат берілмеген. Мекенжай жолағындағы сайт рұқсаттарын ашып, камераны рұқсат етіңіз. VS Code ішіндегі алдын ала қарау орнына кәдімгі Chrome немесе Edge қолданыңыз.",
            NotFoundError: "Браузер камераны таппады. Камера қосылғанын және Windows параметрлерін тексеріңіз.",
            NotReadableError: "Камера ашылмады. OpenCV демосын, Windows Камерасын, Zoom және камераны қолданатын басқа қойындыларды жабыңыз.",
            AbortError: "Камераны қосу үзілді. Басқа камера қолданбаларын жауып, қайта басыңыз.",
            SecurityError: "Браузер камераны бұғаттады. Бетті кәдімгі браузерде http://127.0.0.1:7860 арқылы ашыңыз."
        };
        return (help[error.name] || "Камераны қосу мүмкін болмады. Браузер мен Windows камера рұқсаттарын тексеріңіз.") + " (" + error.name + ")";
    };
    let preview = root?.querySelector("video");
    if (!preview || preview.readyState < 2) {
        let expired = false;
        let timeout;
        const request = navigator.mediaDevices.getUserMedia({video: true, audio: false});
        // If a permission prompt is left unanswered, release any stream
        // eventually granted after our timeout as well.
        request.then(stream => {
            if (expired) stream.getTracks().forEach(track => track.stop());
        }, () => {});
        try {
            const stream = await Promise.race([
                request,
                new Promise((_, reject) => {
                    timeout = setTimeout(() => {
                        expired = true;
                        reject(new DOMException("Permission prompt unanswered", "TimeoutError"));
                    }, 20000);
                })
            ]);
            stream.getTracks().forEach(track => track.stop());
        } catch (error) {
            if (error.name === "TimeoutError") {
                return "Камера рұқсатын күтіп тұрмыз. Браузердегі рұқсат сұрауын тексеріп, «Рұқсат ету» таңдаңыз да, қайта басыңыз.";
            }
            return errorMessage(error);
        } finally {
            clearTimeout(timeout);
        }
        await pause(400);
        const access = [...(root?.querySelectorAll("button") || [])].find(
            button => /webcam|камер/i.test(button.textContent));
        if (!access) {
            return "Камераға рұқсат берілді. Камера панеліндегі қосу батырмасын басыңыз немесе бетті жаңартыңыз.";
        }
        access.click();
        for (let i = 0; i < 40; i++) {
            await pause(200);
            preview = root.querySelector("video");
            if (preview && preview.readyState >= 2) break;
        }
        if (!preview || preview.readyState < 2) {
            return "Рұқсат бар, бірақ камерадан кадр келмеді. Басқа камера қолданбаларын жауып, бетті жаңартыңыз.";
        }
    }
    const start = [...root.querySelectorAll("button")].find(
        button => button.getAttribute("aria-label") === "capture photo");
    if (start && !root.querySelector('[title="stop recording"]')) start.click();
    return "Камера қосылды. Алғашқы нәтижені күтіңіз: модель бірінші рет жүктеледі.";
}
