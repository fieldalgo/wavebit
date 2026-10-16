/* Use PyData's own theme actions, with a one-click light/dark/system control. */
function setupThemeToggle() {
  if (location.hostname.endsWith('.localhost') || location.hostname === 'localhost') {
    const company = new URL(location.origin);
    company.hostname = 'localhost';
    document.querySelectorAll('.wb-company').forEach(link => { link.href = company.href; });
    document.querySelectorAll('.navbar-brand').forEach(link => { link.href = location.origin + '/'; });
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
      toggle.removeAttribute('title');
      toggle.dataset.tooltip = label;
      toggle.dataset.tooltipAlign = 'end';
      toggle.classList.add('site-tooltip');
      try { localStorage.setItem('fieldalgo-theme', mode === 'auto' ? 'system' : mode); } catch {}
    };
    toggle.addEventListener('click', () => {
      const mode = document.documentElement.dataset.mode || 'auto';
      const next = { light: 'dark', dark: 'auto', auto: 'light' }[mode];
      menu.querySelector(`[data-mode="${next}"]`).click();
      updateLabel();
    });
    try {
      const saved = localStorage.getItem('fieldalgo-theme');
      const desired = saved === 'system' ? 'auto' : saved;
      if (['auto', 'light', 'dark'].includes(desired)) menu.querySelector(`[data-mode="${desired}"]`)?.click();
    } catch {}
    new MutationObserver(updateLabel).observe(document.documentElement, {
      attributes: true, attributeFilter: ['data-theme', 'data-mode'],
    });
    updateLabel();
  });
  // Retain the selected language when returning to the product website.
  try { localStorage.setItem('fieldalgo-lang', document.documentElement.lang === 'ko' ? 'KO' : 'EN'); } catch {}
  document.querySelectorAll('.wb-lang').forEach(link => link.addEventListener('click', () => {
    try { localStorage.setItem('fieldalgo-lang', link.lang === 'ko' ? 'KO' : 'EN'); } catch {}
  }));
  // Native title hints and Bootstrap tooltips must not compete on the same controls.
  document.querySelectorAll('.bd-header [data-bs-toggle="tooltip"], .sidebar-header-items [data-bs-toggle="tooltip"], .wb-company, .wb-lang').forEach(control => {
    const theme = control.classList.contains('theme-switch-container');
    const label = control.getAttribute('title') || control.getAttribute('data-bs-original-title') || control.getAttribute('aria-label');
    window.bootstrap?.Tooltip?.getInstance(control)?.dispose();
    control.removeAttribute('title');
    control.removeAttribute('data-bs-toggle');
    control.removeAttribute('data-bs-original-title');
    if (!theme && label) {
      control.dataset.tooltip = label;
      control.dataset.tooltipAlign = 'end';
      control.classList.add('site-tooltip');
      if (!control.hasAttribute('aria-label')) control.setAttribute('aria-label', label);
    }
    control.querySelectorAll('[title]').forEach(child => child.removeAttribute('title'));
  });
}
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', setupThemeToggle);
} else {
  setupThemeToggle();
}
