# Fonts

These are the self-hosted `.woff2` files that `../fonts.css` loads (see DESIGN.md §2). Put these files here:

| File | Family / weight |
|---|---|
| `BarlowCondensed-ExtraLight.woff2` | Barlow Condensed 200 |
| `BarlowCondensed-ExtraBold.woff2` | Barlow Condensed 800 |
| `IBMPlexSans-Regular.woff2` | IBM Plex Sans 400 |
| `IBMPlexSans-Medium.woff2` | IBM Plex Sans 500 |
| `IBMPlexSans-SemiBold.woff2` | IBM Plex Sans 600 |
| `IBMPlexMono-ExtraLight.woff2` | IBM Plex Mono 200 |
| `IBMPlexMono-Regular.woff2` | IBM Plex Mono 400 |
| `IBMPlexMono-Medium.woff2` | IBM Plex Mono 500 |
| `OFL-BarlowCondensed.txt`, `OFL-IBMPlex.txt` | SIL Open Font License texts |

All three families are under the SIL Open Font License 1.1. If a file is missing, the page still works: `font-display: swap` keeps the fallback stack from `../tokens.css`.
