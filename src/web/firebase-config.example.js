// Copy this file to `firebase-config.js` and fill in your project's web config.
// Firebase console -> Project settings -> General -> "Your apps" -> Web app -> SDK setup.
// firebase-config.js is gitignored so your keys are not committed.
export const firebaseConfig = {
  apiKey: "YOUR_API_KEY",
  authDomain: "YOUR_PROJECT.firebaseapp.com",
  projectId: "YOUR_PROJECT_ID",
  storageBucket: "YOUR_PROJECT.appspot.com",
  messagingSenderId: "YOUR_SENDER_ID",
  appId: "YOUR_APP_ID",

  // Optional but recommended — App Check (reCAPTCHA v3) site key. When set,
  // app.js initializes App Check so only your real web app can read Firestore,
  // protecting your read quota from scrapers. Create one at:
  //   Firebase console -> App Check -> Apps -> register with reCAPTCHA v3.
  // Leave as "" to skip App Check. See docs/firebase_setup.md §6.
  recaptchaSiteKey: "",
};
