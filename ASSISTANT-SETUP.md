# Turn on AI answers for your Course Assistant (free, ~10 minutes, no coding)

Your math course already has the assistant built in. Out of the box it answers straight from
your own lessons and glossary — this is **offline mode**: free, private, no account, nothing to
set up, and it keeps working even with no internet after the first load. This guide switches it
into full **AI mode**, where the assistant reads each student's question and writes a real,
reasoned answer grounded in your course — still for free.

You do three things, once: **(1)** get a free Google Gemini key, **(2)** deploy a tiny "Worker"
that keeps that key safe, **(3)** paste the Worker's address into one file — then **lock it to
your site**.

> **This course is already connected.** `assistant-config.js` already contains a Worker URL —
> `https://gd-assistant-proxy.uekpavanharish.workers.dev` — which is the same free Worker shared
> across these courses. If that Worker has a Gemini key set, your students already get AI answers,
> and you can skip straight to **"Lock it down"** below (the one step that protects your quota).
> The full steps are here for when you want to understand it, rebuild it, or run your own Worker.

**Cost: $0.** The Gemini free tier needs no credit card, and Cloudflare's Worker free tier covers
100,000 requests/day. Because no card is attached, **you can never be charged** — worst case, the
free daily quota pauses until the next day and the assistant quietly falls back to offline mode.
**If the Worker is never set up at all, the assistant still works fully in offline mode.**

---

## Step 1 — Get a free Gemini API key (2 min)

1. Go to **https://aistudio.google.com/apikey** and sign in with a Google account.
2. Click **Create API key**.
3. Copy the key (a long string starting with `AIza…`). Keep it handy for Step 3.

No card, no billing.

## Step 2 — Create the free "Worker" (5 min)

1. Go to **https://dash.cloudflare.com** and create a free account (or sign in).
2. Left menu → **Workers & Pages** → **Create** → **Create Worker**.
3. Name it `gd-assistant-proxy`, then click **Deploy** (a placeholder deploys — that's fine).
4. Click **Edit code**. Select everything in the editor and delete it.
5. Paste in the Worker code — the **`worker/assistant-proxy.js`** file that ships with the course
   family. *(This math course points at the already-deployed shared Worker above, so you only need
   your own if you want a separate one; the code is identical either way.)*
6. Click **Deploy** (top right).

## Step 3 — Set the Worker's variables (2 min)

Still in the Worker: **Settings** → **Variables and Secrets**. Add these:

- **Secret** (encrypted) — **Name:** `GEMINI_API_KEY`, **Value:** your key from Step 1. *(Required —
  this is why the key lives in the Worker, never in your public website.)*
- *(Optional)* **Variable** — `GEMINI_MODEL` = `gemini-flash-latest`. The Worker already defaults to
  this (Google's alias for the current free Flash model, so it survives model rotations). Only set it
  if you want to pin or switch the model.
- **Variable** — `ALLOWED_ORIGINS` = `https://saipavan333.github.io`. **Set this — see "Lock it
  down" below.** It restricts which website is allowed to use your Worker.

Click **Save and deploy**.

## Step 4 — Connect it to your site (1 min)

1. At the top of the Worker page, copy its URL — like `https://gd-assistant-proxy.yourname.workers.dev`.
2. Open **`assistant-config.js`** in this course folder (it's in the project **root**, right next to
   `app.js` — not in a subfolder) and paste the URL between the quotes:
   ```js
   window.GD_ASSIST_PROXY = "https://gd-assistant-proxy.yourname.workers.dev";
   ```
3. Save, then publish the change (see **`DEPLOY.md`**).

## Step 5 — Test it

Open your live course, click **Ask AI**, and ask *"what is an eigenvector?"*. You should get a
written, reasoned answer (and, because that term is in the glossary, the definition leads with
`Av = λv`). The dot at the bottom of the panel reads **AI mode**. If the AI is ever unreachable, it
quietly falls back to the offline answer — students never see a raw error.

---

## Lock it down (do this — it protects your free quota)

The Worker's `ALLOWED_ORIGINS` variable decides **which websites are allowed to use it**, and this
is the one setting you should not skip:

- **If you leave `ALLOWED_ORIGINS` blank or unset, the Worker accepts requests from anyone.** Anyone
  who finds the URL can POST questions to it and drain **your** free Gemini quota — which pauses AI
  mode for your real students until the quota resets the next day. (You still can't be charged, but
  your assistant goes dark.)
- **Set it to your site's address and only your course can use the Worker.** The Worker checks the
  browser's `Origin` against your list and rejects everything else.

**For this course, set:**

```
ALLOWED_ORIGINS = https://saipavan333.github.io
```

**How that value was determined:** this site is published from the GitHub repo
`saipavan333/mathematics-for-ai` (a GitHub Pages **project site**), so it lives at
`https://saipavan333.github.io/mathematics-for-ai/`. An "origin" is only the scheme **+** domain —
**never the path** — so the value is `https://saipavan333.github.io` with **no
`/mathematics-for-ai`, and no trailing slash**. It must match exactly.

- **Format:** `https://<your-github-username>.github.io`. Use a comma-separated list if you serve
  the course from more than one address — for example, adding a custom domain:
  `https://saipavan333.github.io,https://yourdomain.com`.
- **Where to find yours:** in your repo on GitHub, open **Settings → Pages**; it shows your live
  URL. Your origin is the part up to (but not including) the first `/` after `https://`.
- Any other course you publish under this same GitHub account shares the `https://saipavan333.github.io`
  origin, so this one value covers them all.
- After changing it, click **Save and deploy** on the Worker.

---

## Troubleshooting

- **Still says "Offline mode":** confirm `assistant-config.js` has the Worker URL and you published
  it, then hard-refresh the page (Ctrl+Shift+R).
- **"Couldn't get an AI answer" / "Origin not allowed":** if you just set `ALLOWED_ORIGINS`, make
  sure it matches your site's origin **exactly** (no trailing slash, no path) and re-deploy. Also
  confirm `GEMINI_API_KEY` is set in **Settings → Variables and Secrets**.
- **"This model is no longer available":** Google occasionally retires model names. The Worker
  already defaults to `gemini-flash-latest`, which auto-tracks the current model. If you ever see
  this, add/confirm a **Variable** `GEMINI_MODEL` = `gemini-flash-latest`, then Save and deploy.
- **Rate-limit errors when many students use it at once:** switch `GEMINI_MODEL` to a lighter Flash
  model for a higher free daily limit — check the current model names at
  https://aistudio.google.com/apikey.

## Optional: see what students are asking (5 min)

The Worker can log the questions students type — gold for spotting gaps in your course. It's off
until you turn it on:

1. **Create a place to store them.** Left menu → **Storage & Databases** → **KV** → **Create a
   namespace**, name it `questions`.
2. **Connect it to the Worker.** Worker → **Settings** → **Bindings** → **Add** → **KV namespace**:
   Variable name `QLOG`, and pick the `questions` namespace.
3. **Set a password for the view page.** Worker → **Settings** → **Variables and Secrets** → add a
   **Secret** named `ADMIN_TOKEN` with any password you choose.
4. **Save and deploy.**
5. **View them anytime** at: `https://gd-assistant-proxy.yourname.workers.dev/?admin=YOUR_ADMIN_TOKEN`

(Questions auto-expire after 90 days. Leave `QLOG` unbound and nothing is logged.)

## Later, when you charge for the course

Right now anyone on your site can use the assistant. When you add student logins (subscriptions),
the Worker can be extended to require a valid login token, so only paying students use your quota.
