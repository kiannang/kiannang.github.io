import Link from "next/link";
import { Github, Linkedin, Mail } from "lucide-react";
import { site } from "@/data/site";
import { Container } from "@/components/ui/container";
import { Reveal } from "@/components/ui/reveal";

export function ContactSection() {
  return (
    <Reveal>
      <Container id="contact" className="px-6 py-10 sm:px-8 sm:py-14 lg:px-12 lg:py-20">
        <div className="grid gap-12 lg:grid-cols-[1.2fr_0.8fr] lg:items-center">

          <div className="space-y-6">
            <p className="font-mono text-xs uppercase tracking-[0.28em] text-accent">/ contact</p>
            <h2 className="leading-none">
              <span className="block text-5xl font-black text-foreground sm:text-6xl lg:text-7xl">
                Nice to
              </span>
              <span className="block font-display italic text-6xl text-foreground sm:text-7xl lg:text-8xl">
                Meet You.
              </span>
            </h2>
            <p className="max-w-md text-lg leading-8 text-muted">
              Interested in collaboration or learning more about my work? I&apos;d love to hear from you.
            </p>

            <div className="flex flex-wrap gap-3 pt-2">
              <Link
                href={`mailto:${site.email}`}
                className="inline-flex items-center gap-2 rounded-full bg-accent px-5 py-3 text-sm font-semibold text-[#f4c9d6] hover:bg-accent/80"
              >
                <Mail className="h-4 w-4" />
                Email Me
              </Link>
              <Link
                href={site.links.linkedin}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-2 rounded-full border border-accent/30 bg-accentSoft px-5 py-3 text-sm font-medium text-foreground hover:border-accent/60"
              >
                <Linkedin className="h-4 w-4" />
                LinkedIn
              </Link>
              <Link
                href={site.links.github}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-2 rounded-full border border-line px-5 py-3 text-sm font-medium text-muted hover:border-accent/30 hover:text-foreground"
              >
                <Github className="h-4 w-4" />
                GitHub
              </Link>
            </div>
          </div>

          <div className="rounded-[32px] border border-accent/20 bg-accentSoft p-8 space-y-4">
            <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted">Reach me at</p>
            <p className="text-xl font-semibold text-foreground">{site.email}</p>
            <div className="pt-2">
              <Link
                href="/resume/CV.pdf"
                className="inline-flex items-center gap-2 rounded-full border border-accent/30 px-5 py-3 text-sm font-medium text-foreground hover:border-accent/60 hover:bg-accentSoft"
              >
                View Full CV
              </Link>
            </div>
          </div>

        </div>
      </Container>
    </Reveal>
  );
}
