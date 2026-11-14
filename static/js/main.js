/**
 * Virtual Judge frontend script
 *
 * Handles:
 * - Asynchronous form submissions (two-sided dispute creation, argument submission)
 * - Live word counting for argument inputs
 * - Lightweight animated metric counters
 */

async function postForm(url, payload) {
    const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!response.ok) {
        throw new Error(data.error || data.message || "Request failed");
    }
    return data;
}

function setupWordCounters() {
    const bindCounter = (textareaId, counterId) => {
        const area = document.getElementById(textareaId);
        const count = document.getElementById(counterId);
        if (!area || !count) return;

        const update = () => {
            const words = area.value.trim() ? area.value.trim().split(/\s+/).length : 0;
            count.textContent = `${words} word${words === 1 ? "" : "s"}`;
        };
        area.addEventListener("input", update);
        update();
    };

    bindCounter("argumentTextA", "wordCountA");
    bindCounter("argumentTextB", "wordCountB");
    bindCounter("argumentText", "wordCount");
}

function setupDisputeForms() {
    // Two-sided dispute creation form
    const createForm = document.getElementById("createDisputeForm");
    if (createForm) {
        createForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const submitBtn = document.getElementById("submitBtn");
            const originalText = submitBtn ? submitBtn.innerHTML : "";

            try {
                if (submitBtn) {
                    submitBtn.disabled = true;
                    submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Evaluating with Virtual Judge...';
                }

                const payload = {
                    title: (createForm.title.value || "").trim(),
                    description: (createForm.description ? createForm.description.value : "").trim(),
                    user_a_name: (createForm.user_a_name ? createForm.user_a_name.value : "Side A").trim(),
                    argument_text_a: (createForm.argument_text_a ? createForm.argument_text_a.value : "").trim(),
                    user_b_name: (createForm.user_b_name ? createForm.user_b_name.value : "Side B").trim(),
                    argument_text_b: (createForm.argument_text_b ? createForm.argument_text_b.value : "").trim()
                };

                const data = await postForm("/dispute/create", payload);
                if (data.redirect_url) {
                    window.location.href = data.redirect_url;
                } else {
                    window.location.href = "/disputes";
                }
            } catch (err) {
                alert(err.message || "Failed to submit dispute");
                if (submitBtn) {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = originalText;
                }
            }
        });
    }

    // Opposing argument submission on waiting dispute
    const submitArgForm = document.getElementById("submitArgumentForm");
    if (submitArgForm) {
        submitArgForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const disputeId = submitArgForm.getAttribute("data-dispute-id");
            const submitBtn = document.getElementById("submitArgumentBtn");
            const originalText = submitBtn ? submitBtn.innerHTML : "";

            try {
                if (submitBtn) {
                    submitBtn.disabled = true;
                    submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Analyzing arguments...';
                }

                const payload = {
                    user_name: (submitArgForm.user_name ? submitArgForm.user_name.value : "").trim(),
                    argument_text: (submitArgForm.argument_text ? submitArgForm.argument_text.value : "").trim()
                };

                const data = await postForm(`/dispute/${disputeId}/submit`, payload);
                if (data.redirect_url) {
                    window.location.href = data.redirect_url;
                } else {
                    window.location.reload();
                }
            } catch (err) {
                alert(err.message || "Failed to submit argument");
                if (submitBtn) {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = originalText;
                }
            }
        });
    }
}

function setupStatCounters() {
    const elements = Array.from(document.querySelectorAll(".stat-counter"));
    if (!elements.length) return;

    elements.forEach((el) => {
        const target = parseFloat(el.getAttribute("data-target") || "0");
        if (Number.isNaN(target)) return;

        let current = 0;
        const duration = 800;
        const start = performance.now();

        const animate = (now) => {
            const progress = Math.min((now - start) / duration, 1);
            const value = Math.round(progress * target);
            el.textContent = String(value);
            if (progress < 1) {
                requestAnimationFrame(animate);
            } else {
                el.textContent = String(target);
            }
        };
        requestAnimationFrame(animate);
    });
}

function setupProgressBars() {
    document.querySelectorAll(".progress-bar[data-width]").forEach((el) => {
        const val = el.getAttribute("data-width") || "50";
        el.style.width = val + "%";
    });
}

document.addEventListener("DOMContentLoaded", () => {
    setupWordCounters();
    setupDisputeForms();
    setupStatCounters();
    setupProgressBars();
});
