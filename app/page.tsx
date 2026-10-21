import { ContactSection } from "@/components/contact-section";
import { ExperienceTimeline } from "@/components/experience-timeline";
import { EducationAwards } from "@/components/education-awards";
import { Footer } from "@/components/footer";
import { Hero } from "@/components/hero";
import { Navbar } from "@/components/navbar";
import { PublicationCard } from "@/components/publication-card";
import { SkillsGrid } from "@/components/skills-grid";
import { TeachingSection } from "@/components/teaching-section";

export default function HomePage() {
  return (
    <div className="min-h-screen">
      <Navbar />

      <main className="mx-auto w-full max-w-[800px] px-4">
        <Hero />
        <EducationAwards />
        <PublicationCard />
        <ExperienceTimeline />
        <TeachingSection />
        <SkillsGrid />
        <ContactSection />
      </main>

      <Footer />
    </div>
  );
}
