# Kianna Ng Portfolio

A portfolio and blog built with Next.js App Router, TypeScript, Tailwind CSS, Framer Motion, and Lucide React.

## Stack

- Next.js App Router
- TypeScript
- Tailwind CSS
- Framer Motion
- Lucide React

## Project Structure

```text
app/
  blog/
  globals.css
  layout.tsx
  page.tsx
components/
  contact-section.tsx
  current-focus.tsx
  experience-timeline.tsx
  footer.tsx
  hero.tsx
  marquee.tsx
  navbar.tsx
  publication-card.tsx
  research-areas.tsx
  section-header.tsx
  selected-work.tsx
  share-button.tsx
  skills-grid.tsx
  teaching-section.tsx
  ui/
data/
  experience.ts
  focus.ts
  publications.ts
  researchAreas.ts
  selectedWork.ts
  site.ts
  skills.ts
  teaching.ts
lib/
  posts.ts
  utils.ts
posts/
  *.md
public/
  images/
  resume/
```

## Getting Started

1. Install dependencies:

```bash
npm install
```

2. Run the development server:

```bash
npm run dev
```

3. Open `http://localhost:3000`.

## Build

```bash
npm run build
npm run start
```

## Deploy to Vercel

1. Push the repository to GitHub.
2. Import the project into Vercel.
3. Use the default Next.js settings.
4. Deploy.

## Deploy to GitHub Pages

This repo is configured for static export and can be deployed using only GitHub.

1. Push the repository to GitHub.
2. In GitHub, open `Settings` -> `Pages`.
3. Under `Build and deployment`, set `Source` to `GitHub Actions`.
4. Push to `main` and GitHub will build and publish the site automatically.

The workflow is defined in `.github/workflows/deploy-pages.yml` and publishes the generated `out/` directory.

## Customization Notes

- The live CV is served from `public/resume/CV.pdf`.
- Update links and profile metadata in `data/site.ts`.
- Blog posts live in `posts/` and are rendered through `lib/posts.ts`.
- Edit the content in the `data/` directory without touching the layout components.
