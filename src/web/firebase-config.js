// Firebase Web SDK config for the Business Activity - Live Footfall dashboard.
//
// These values are the PUBLIC identifier of the Firebase project. They ship in
// every visitor's browser and are safe to commit — see the Firebase docs:
// https://firebase.google.com/docs/projects/api-keys. The real protection is
// firestore.rules (read-only for the three dashboard collections, all writes
// denied) plus the Admin SDK service-account key, which is gitignored and
// stays on the admin's machine only.
export const firebaseConfig = {
  apiKey: "AIzaSyDUVA5-SJ0v_8PsUfHKo8sQ1B_dcZh4hD0",
  authDomain: "turkey-footfall.firebaseapp.com",
  projectId: "turkey-footfall",
  storageBucket: "turkey-footfall.firebasestorage.app",
  messagingSenderId: "457687660625",
  appId: "1:457687660625:web:b72053ddd96aee5f9de070",

  // Optional — App Check (reCAPTCHA v3) site key. When set, app.js initializes
  // App Check so only your real web app can read Firestore, protecting the read
  // quota from scrapers. Leave "" to skip. See docs/firebase_setup.md §6.
  recaptchaSiteKey: "",
};
