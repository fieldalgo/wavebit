/* Use PyData's own theme actions, with a one-click light/dark control. */
function setupThemeToggle() {
  if (location.hostname.endsWith('.localhost') || location.hostname === 'localhost') {
    const company = new URL(location.origin);
    company.hostname = 'localhost';
    document.querySelectorAll('.wb-company').forEach(link => { link.href = company.href; });
  }
  document.querySelectorAll('.theme-switch-container').forEach(container => {
    const toggle = container.querySelector('.theme-switch-button');
    const menu = container.querySelector('.dropdown-menu');
    if (!toggle || !menu) return;
    toggle.removeAttribute('data-bs-toggle');
    toggle.classList.remove('dropdown-toggle');
    menu.hidden = true;
    const updateLabel = () => {
      const dark = document.documentElement.dataset.theme === 'dark';
      const korean = document.documentElement.lang === 'ko';
      const label = korean
        ? (dark ? '라이트 모드로 전환' : '다크 모드로 전환')
        : (dark ? 'Switch to light mode' : 'Switch to dark mode');
      toggle.setAttribute('aria-label', label);
      toggle.title = label;
    };
    toggle.addEventListener('click', () => {
      const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
      menu.querySelector(`[data-mode="${next}"]`).click();
      updateLabel();
    });
    new MutationObserver(updateLabel).observe(document.documentElement, {
      attributes: true, attributeFilter: ['data-theme'],
    });
    updateLabel();
  });
}
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', setupThemeToggle);
} else {
  setupThemeToggle();
}
