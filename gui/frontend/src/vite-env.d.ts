/// <reference types="vite/client" />

// Figma Make asset imports ("figma:asset/<hash>.png") resolve to files under
// src/assets/ via the alias map in vite.config.ts; for the type-checker they
// are plain image URLs.
declare module 'figma:asset/*' {
  const src: string;
  export default src;
}
