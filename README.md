# Kianna Ng Portfolio

A dark, research-forward portfolio site built with Next.js App Router, TypeScript, Tailwind CSS, Framer Motion, and Lucide React.

## Stack

- Next.js App Router
- TypeScript
- Tailwind CSS
- Framer Motion
- Lucide React

## Project Structure

```text
app/
  globals.css
  layout.tsx
  page.tsx
components/
  contact-section.tsx
  current-focus.tsx
  experience-timeline.tsx
  footer.tsx
  hero.tsx
  navbar.tsx
  publication-card.tsx
  research-areas.tsx
  section-header.tsx
  selected-work.tsx
  skills-grid.tsx
  ui/
data/
  experience.ts
  focus.ts
  publications.ts
  researchAreas.ts
  selectedWork.ts
  site.ts
  skills.ts
lib/
  utils.ts
public/
  resume.pdf
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

- Update the CV file path in the components if you change the current `resume/CV.pdf` location.
- Update links and profile metadata in `data/site.ts`.
- If you want to expose an email address publicly, update `components/contact-section.tsx`.
- Edit the content in the `data/` directory without touching the layout components.
