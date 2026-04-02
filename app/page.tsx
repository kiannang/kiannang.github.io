import { ContactSection } from "@/components/contact-section";
import { ExperienceTimeline } from "@/components/experience-timeline";
import { Footer } from "@/components/footer";
import { Hero } from "@/components/hero";
import { Marquee } from "@/components/marquee";
import { Navbar } from "@/components/navbar";
import { PublicationCard } from "@/components/publication-card";
import { SkillsGrid } from "@/components/skills-grid";
import { TeachingSection } from "@/components/teaching-section";

export default function HomePage() {
  return (
    <div className="relative min-h-screen overflow-x-clip">
      <div className="pointer-events-none absolute inset-0 -z-10">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(244,201,214,0.12),transparent_40%),radial-gradient(circle_at_bottom_left,rgba(244,201,214,0.08),transparent_40%)]" />
      </div>

      <Navbar />

      <main className="mx-auto flex w-full max-w-7xl flex-col gap-0 px-0 pb-0 pt-24">
        <Hero />
      </main>

      <Marquee />

      <main className="mx-auto flex w-full max-w-7xl flex-col gap-8 px-4 py-8 sm:px-6 lg:px-8">
        <PublicationCard />
        <ExperienceTimeline />
        <TeachingSection />
        <SkillsGrid />
      </main>

      <Marquee />

      <main className="mx-auto flex w-full max-w-7xl flex-col gap-8 px-4 py-8 sm:px-6 lg:px-8">
        <ContactSection />
      </main>

      <Footer />
    </div>
  );
}
