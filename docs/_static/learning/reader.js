(()=>{
  'use strict';
  const article = document.querySelector('.bd-article');
  if (!article) return;
  article.querySelectorAll('div.highlight').forEach(block => {
    const pre = block.querySelector('pre');
    if (!pre || !navigator.clipboard) return;
    const button = document.createElement('button');
    button.className = 'code-copy'; button.type = 'button'; button.textContent = 'Copy code';
    button.addEventListener('click', async () => {
      try {
        const code = pre.cloneNode(true);
        code.querySelectorAll('.linenos').forEach(line => line.remove());
        await navigator.clipboard.writeText(code.textContent);
        button.textContent = 'Copied';
      }
      catch { button.textContent = 'Select the code and copy it'; }
      setTimeout(() => { button.textContent = 'Copy code'; }, 2000);
    });
    block.before(button);
  });
  const figures = article.querySelectorAll('.learning-figure img');
  if (!figures.length) return;
  const dialog = document.createElement('dialog');
  dialog.className = 'image-viewer'; dialog.setAttribute('aria-label', 'Enlarge figure');
  const close = document.createElement('button'); close.textContent = 'Close'; close.type = 'button';
  const scroll = document.createElement('div'); scroll.className = 'image-scroll';
  const image = document.createElement('img'); const caption = document.createElement('div');
  caption.className = 'image-caption';
  scroll.append(image); dialog.append(close,scroll,caption); document.body.append(dialog);
  close.addEventListener('click',()=>dialog.close());
  dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
  figures.forEach(figure => {
    figure.tabIndex = 0; figure.setAttribute('role','button');
    figure.setAttribute('aria-label', `${figure.alt} — Enlarge`);
    const open = () => {
      image.src = figure.src; image.alt = figure.alt;
      const sourceCaption = figure.closest('figure')?.querySelector('figcaption');
      caption.replaceChildren();
      caption.hidden = !sourceCaption;
      if (sourceCaption) {
        const content = sourceCaption.cloneNode(true);
        content.querySelectorAll('.headerlink').forEach(link => link.remove());
        content.querySelectorAll('[id]').forEach(node => node.removeAttribute('id'));
        caption.append(...content.childNodes);
      }
      dialog.showModal();
    };
    figure.addEventListener('click',open);
    figure.addEventListener('keydown', event => { if (event.key==='Enter'||event.key===' ') { event.preventDefault();open(); } });
  });
})();
