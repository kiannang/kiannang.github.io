import { ContactSection } from "@/components/contact-section";
import { CurrentFocus } from "@/components/current-focus";
import { ExperienceTimeline } from "@/components/experience-timeline";
import { Footer } from "@/components/footer";
import { Hero } from "@/components/hero";
import { Navbar } from "@/components/navbar";
import { PublicationCard } from "@/components/publication-card";
import { ResearchAreas } from "@/components/research-areas";
import { SelectedWork } from "@/components/selected-work";
import { SkillsGrid } from "@/components/skills-grid";

export default function HomePage() {
  return (
    <div className="relative min-h-screen overflow-x-clip">
      <div className="pointer-events-none absolute inset-0 -z-10">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(76,171,255,0.16),transparent_32%),radial-gradient(circle_at_80%_20%,rgba(121,111,255,0.14),transparent_26%),linear-gradient(180deg,#05070b_0%,#090d14_42%,#05070b_100%)]" />
        <div className="absolute inset-0 bg-grid bg-[size:44px_44px] opacity-[0.08]" />
        <div className="absolute left-1/2 top-0 h-[34rem] w-[34rem] -translate-x-1/2 rounded-full bg-accent/10 blur-3xl" />
      </div>

      <Navbar />

      <main className="mx-auto flex w-full max-w-7xl flex-col gap-8 px-4 pb-10 pt-24 sm:px-6 lg:px-8">
        <Hero />
        <CurrentFocus />
        <ResearchAreas />
        <SelectedWork />
        <PublicationCard />
        <ExperienceTimeline />
        <SkillsGrid />
        <ContactSection />
      </main>

      <Footer />
    </div>
  );
}
