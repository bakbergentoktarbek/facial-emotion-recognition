() => {
    // Gradio 5's webcam-permission caption is hard-coded in English. Localize
    // visible text without replacing controls or their capture event handlers.
    const words = new Map([
        ["Click to Access Webcam", "Камераға қолжетімділік беру"],
        ["Drop Image Here", "Суретті осында сүйреп әкеліңіз"],
        ["Drop image here", "Суретті осында сүйреп әкеліңіз"],
        ["Перетащите изображение сюда", "Суретті осында сүйреп әкеліңіз"],
        ["- или -", "немесе"], ["- or -", "немесе"],
        ["Click to Upload", "Жүктеу үшін басыңыз"],
        ["Нажмите для загрузки", "Жүктеу үшін басыңыз"],
        ["Record", "Ағынды бастау"], ["Запись", "Ағынды бастау"],
        ["Stop", "Тоқтату"], ["Стоп", "Тоқтату"],
        ["Waiting", "Күтіңіз…"], ["Waiting...", "Күтіңіз…"],
    ]);
    const translate = node => {
        if (node.nodeType === Node.TEXT_NODE) {
            if (["SCRIPT", "STYLE"].includes(node.parentElement?.tagName)) return;
            const value = words.get(node.textContent.trim());
            if (value) node.textContent = value;
            return;
        }
        if (node.nodeType !== Node.ELEMENT_NODE) return;
        const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT);
        const texts = [];
        while (walker.nextNode()) texts.push(walker.currentNode);
        texts.forEach(translate);
    };
    const app = document.querySelector(".gradio-container");
    if (!app) return;
    // Keep the full-width dark header aligned with the actual header/nav
    // height at each viewport size. CSS handles tab selection immediately.
    const header = document.getElementById("app-header");
    const navigation = document.querySelector("#workspace > .tab-wrapper");
    const syncHeader = () => {
        const bottom = Math.max(header?.getBoundingClientRect().bottom || 0,
                                navigation?.getBoundingClientRect().bottom || 0);
        document.documentElement.style.setProperty("--mimika-header-bottom",
            Math.ceil(bottom + window.scrollY + 16) + "px");
    };
    window.mimikaHeaderLayout?.disconnect();
    window.mimikaHeaderLayout = new ResizeObserver(syncHeader);
    if (header) window.mimikaHeaderLayout.observe(header);
    if (navigation) window.mimikaHeaderLayout.observe(navigation);
    requestAnimationFrame(syncHeader);
    translate(app);
    window.mimikaTranslations?.disconnect();
    window.mimikaTranslations = new MutationObserver(changes => {
        for (const change of changes) {
            if (change.type === "characterData") translate(change.target);
            else change.addedNodes.forEach(translate);
        }
    });
    window.mimikaTranslations.observe(app, {subtree: true, childList: true, characterData: true});
}
