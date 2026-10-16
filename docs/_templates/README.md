# Documentation templates

| Template | Question it answers |
| --- | --- |
| `language-switcher.html` | How does a reader switch the current page between the English root and Korean subtree? |

`docs/conf.py` supplies relative links only when the bilingual HTML build is
explicitly enabled. Single-language builds expose no language links.

## Product-site styling

`../_static/wavebit-site.css` is the documentation's product-style layer. It is
loaded after `learning/learning.css` by `docs/conf.py`; keep that ordering when
changing the theme. It adjusts typography, colors, header spacing and cards while
retaining PyData's search, sidebars, language links and code controls.

Build documentation normally to include this stylesheet in both language trees.
For a CSS-only preview against an existing bilingual build, replace
`_static/wavebit-site.css` and `ko/_static/wavebit-site.css` with the source file;
HTML and the search index do not need to be regenerated. Check light and dark
modes, desktop and mobile navigation, and a page containing a code block.
