# Publishing this course to the web (GitHub Pages)

The whole app is static files, so hosting it is free and takes a few minutes. Once it's up
you'll have a URL like `https://YOURNAME.github.io/math-for-ai/` that works on any device —
laptop, tablet, phone. This folder already includes a `.nojekyll` file so GitHub serves the
files as-is.

Pick **one** of the two methods below. Method A needs no tools; Method B is nicer for
pushing updates later.

---

## Method A — Upload in the browser (no installs, easiest)

1. **Make a GitHub account** (free): go to <https://github.com> and sign up. Note your
   username — it becomes part of your site's address.

2. **Create a repository.** Top-right **+** → **New repository**.
   - **Repository name:** `math-for-ai` (or anything you like).
   - **Visibility:** **Public** (required for free GitHub Pages).
   - Leave everything else default and click **Create repository**.

3. **Upload the files.** On the new repo's page, click **uploading an existing file**
   (or **Add file → Upload files**). Then, from this `Mathematics` folder, select **all**
   of these and drag them into the browser:
   - `index.html`, `styles.css`, `app.js`, `curriculum.js`
   - the entire **`content`** folder
   - `.nojekyll`  *(if you don't see it, turn on "show hidden files" in your file explorer;
     it's important)*
   - the `.md` files are optional but nice to include
   Click **Commit changes**.

4. **Turn on Pages.** In the repo, go to **Settings** (top tab) → **Pages** (left sidebar,
   under "Code and automation").
   - Under **Build and deployment → Source**, choose **Deploy from a branch**.
   - **Branch:** `main`, folder **`/ (root)`**, then **Save**.

5. **Visit your site.** Wait a few minutes (up to ~10). Refresh the Pages settings screen
   and the URL appears at the top: **`https://YOURNAME.github.io/math-for-ai/`**. Open it —
   that's your course, live and shareable. Bookmark it on your phone.

**To update later:** repo → **Add file → Upload files** → drag the changed file(s) →
**Commit**. The site refreshes within a few minutes.

---

## Method B — Git command line (best for frequent updates)

Requires [Git](https://git-scm.com/downloads) installed. Run these **from inside this
`Mathematics` folder**:

```bash
git init
git add .
git commit -m "Mathematics for AI Engineers — initial site"
git branch -M main
```

Create an empty **public** repo on GitHub named `math-for-ai` (don't add a README there),
then connect and push (replace `YOURNAME`):

```bash
git remote add origin https://github.com/YOURNAME/math-for-ai.git
git push -u origin main
```

Then do **step 4** above (Settings → Pages → Deploy from a branch → `main` / `root` → Save).

**To update later:**

```bash
git add .
git commit -m "add a new track"
git push
```

The live site rebuilds automatically within a few minutes.

---

## Good to know

- **It's public.** GitHub Pages sites are visible to anyone with the link, even though
  there's nothing sensitive here. Don't commit private files.
- **First load needs internet.** The math (KaTeX) and the Python runtime (Pyodide) come
  from a CDN, so the very first visit on a device needs a connection; after that the
  browser caches them.
- **Custom domain (optional).** If you own a domain, Settings → Pages → "Custom domain"
  lets you use it instead of the `github.io` address.
- **A shorter URL.** If you name the repository exactly `YOURNAME.github.io`, the site is
  served at `https://YOURNAME.github.io/` (no `/math-for-ai` suffix). Either works.
