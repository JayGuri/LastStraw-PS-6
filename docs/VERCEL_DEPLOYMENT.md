# Vercel deployment – fixing “redirect to localhost” on sign-in

If Google (or other) sign-in redirects you to **localhost** instead of your Vercel app, the app is still using local URLs. Fix it by setting the right environment variables in **both** the frontend (Vercel) and the backend (wherever it runs).

---

## 1. Frontend (Vercel)

In **Vercel** → your project → **Settings** → **Environment Variables**, add:

| Variable         | Value (example)                    | Notes |
|------------------|-------------------------------------|--------|
| `VITE_API_URL`   | `https://your-backend.com`          | **Required.** Your backend base URL (no trailing slash). Used for login and API calls. If this is wrong or missing, “Sign in with Google” can send users to localhost. |

Other optional frontend vars (if you use them): `VITE_MOCK_MODE`, `VITE_CESIUM_TOKEN`, `VITE_RAZORPAY_ID`, `VITE_RAZORPAY_KEY`.

After changing env vars, **redeploy** the frontend (Vite bakes `VITE_*` into the build at deploy time).

---

## 2. Backend (Railway / Render / Vercel serverless / etc.)

Wherever the **mongo** FastAPI backend is hosted, set:

| Variable               | Value (example)                                      | Notes |
|------------------------|------------------------------------------------------|--------|
| `FRONTEND_URL`         | `https://your-app.vercel.app`                        | **Required.** Exact URL of your Vercel app (no trailing slash). After OAuth, the backend redirects the user here with `?token=...`. If this is localhost, users land on localhost. |
| `GOOGLE_REDIRECT_URI`  | `https://your-backend.com/auth/google/callback`      | **Required for Google sign-in.** Must be the **backend** callback URL (not the frontend). Must match what you add in Google Cloud Console (see below). |

Optional: `CORS_ORIGINS` – comma-separated extra origins if you need more than `FRONTEND_URL`.

---

## 3. Google Cloud Console (Google OAuth only)

1. Open [Google Cloud Console](https://console.cloud.google.com/) → your project → **APIs & Services** → **Credentials**.
2. Edit your **OAuth 2.0 Client ID** (Web application).
3. Under **Authorized redirect URIs**, add:
   - `https://your-backend.com/auth/google/callback`  
   (use the same URL as `GOOGLE_REDIRECT_URI`).
4. Under **Authorized JavaScript origins**, add:
   - `https://your-app.vercel.app`  
   (your Vercel frontend URL).
5. Save.

---

## Quick checklist

- [ ] **Vercel (frontend):** `VITE_API_URL` = your backend base URL → redeploy.
- [ ] **Backend:** `FRONTEND_URL` = your Vercel app URL.
- [ ] **Backend:** `GOOGLE_REDIRECT_URI` = `https://<backend-host>/auth/google/callback`.
- [ ] **Google Console:** Redirect URI and JS origin set as above.

After this, sign-in should redirect back to your Vercel app instead of localhost.
