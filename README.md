# zachparent-site

Astro site scaffolded with Cloudflare C3 for Cloudflare Pages.

## Commands

Run commands from the repository root:

| Command | Action |
| --- | --- |
| `npm install` | Install dependencies |
| `npm run dev` | Start the Astro dev server |
| `npm run build` | Build the site to `dist/` |
| `npm run preview` | Build and preview with Wrangler Pages locally |
| `npm run deploy` | Build and deploy `dist/` to the `zachparent-site` Cloudflare Pages project |
| `npm run cf-typegen` | Regenerate Cloudflare runtime types |

## Cloudflare Pages

This repo is configured for the Cloudflare Pages project `zachparent-site`.

The GitHub Actions workflow in `.github/workflows/deploy-cloudflare-pages.yml` builds on pushes and pull requests targeting `main`, then deploys with `cloudflare/wrangler-action`.

Required GitHub repository secrets:

| Secret | Value |
| --- | --- |
| `CLOUDFLARE_ACCOUNT_ID` | Cloudflare account ID |
| `CLOUDFLARE_API_TOKEN` | API token with Account > Cloudflare Pages > Edit permission |

The Pages build output directory is configured in `wrangler.jsonc` as `./dist`.
