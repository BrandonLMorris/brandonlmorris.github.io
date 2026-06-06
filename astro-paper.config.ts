import { defineAstroPaperConfig } from "./src/types/config";

export default defineAstroPaperConfig({
  site: {
    url: "https://brandonmorris.dev/",
    title: "Brandon Morris",
    description:
      "The personal and research blog of Brandon L. Morris — software engineer working in automated mobile testing, machine learning, and software infrastructure.",
    author: "Brandon L. Morris",
    profile: "https://github.com/brandonlmorris",
    ogImage: "default-og.jpg",
    lang: "en",
    timezone: "America/Los_Angeles",
    dir: "ltr",
  },
  posts: {
    perPage: 5,
    perIndex: 5,
    scheduledPostMargin: 15 * 60 * 1000,
  },
  features: {
    lightAndDarkMode: true,
    dynamicOgImage: false,
    showArchives: true,
    showBackButton: true,
    editPost: {
      enabled: false,
    },
    search: "pagefind",
  },
  socials: [
    { name: "github", url: "https://github.com/brandonlmorris" },
    { name: "x", url: "https://x.com/blancemorris" },
    { name: "mail", url: "mailto:mail@brandonmorris.dev" },
  ],
  shareLinks: [
    { name: "x", url: "https://x.com/intent/post?url=" },
    { name: "mail", url: "mailto:?subject=See%20this%20post&body=" },
  ],
});
