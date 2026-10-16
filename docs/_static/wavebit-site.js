/* Use PyData's own theme actions, with a one-click light/dark/system control. */
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
      const mode = document.documentElement.dataset.mode || 'auto';
      const next = { light: 'dark', dark: 'auto', auto: 'light' }[mode];
      const korean = document.documentElement.lang === 'ko';
      const names = korean ? { light: '라이트', dark: '다크', auto: '시스템' } : { light: 'Light', dark: 'Dark', auto: 'System' };
      const label = korean
        ? `현재 ${names[mode]} 모드 · ${names[next]} 모드로 전환`
        : `Current: ${names[mode]} · Switch to ${names[next]}`;
      toggle.setAttribute('aria-label', label);
      toggle.title = label;
    };
    toggle.addEventListener('click', () => {
      const mode = document.documentElement.dataset.mode || 'auto';
      const next = { light: 'dark', dark: 'auto', auto: 'light' }[mode];
      menu.querySelector(`[data-mode="${next}"]`).click();
      updateLabel();
    });
    new MutationObserver(updateLabel).observe(document.documentElement, {
      attributes: true, attributeFilter: ['data-theme', 'data-mode'],
    });
    updateLabel();
  });
}
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', setupThemeToggle);
} else {
  setupThemeToggle();
}
