import Link from "next/link";
import { Github, Linkedin, Mail } from "lucide-react";
import { site } from "@/data/site";
import { Container } from "@/components/ui/container";
import { Reveal } from "@/components/ui/reveal";

export function ContactSection() {
  return (
    <Reveal>
      <Container id="contact" className="px-6 py-8 sm:px-8 sm:py-10 lg:px-10">
        <div className="grid gap-8 lg:grid-cols-[1.1fr_0.9fr] lg:items-center">
          <div className="space-y-5">
            <p className="font-mono text-xs uppercase tracking-[0.28em] text-accent">/ contact</p>
            <h2 className="text-3xl font-semibold tracking-tight text-foreground sm:text-4xl">Let&apos;s connect.</h2>
            <p className="max-w-2xl text-base leading-8 text-muted sm:text-lg">
              Interested in collaboration, research conversations, or learning more about my work? Feel free to reach
              out. Some current projects are not yet public, but I&apos;m always happy to discuss my interests,
              background, and experience.
            </p>
          </div>

          <div className="rounded-[28px] border border-line bg-white/[0.03] p-6">
            <div className="grid gap-3">
              <div className="rounded-[22px] border border-line bg-panelStrong p-4">
                <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted">Primary contact</p>
                <p className="mt-2 text-sm leading-7 text-slate-200">
                  Direct email is intentionally omitted from the public site. Please connect via LinkedIn or GitHub,
                  and additional contact details can be shared when appropriate.
                </p>
              </div>

              <div className="flex flex-wrap gap-3">
                <button
                  type="button"
                  aria-disabled="true"
                  className="inline-flex cursor-not-allowed items-center gap-2 rounded-full border border-line bg-white/[0.03] px-4 py-3 text-sm text-muted opacity-80"
                >
                  <Mail className="h-4 w-4" />
                  Email on request
                </button>
                <Link
                  href={site.links.github}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-2 rounded-full border border-line px-4 py-3 text-sm text-muted hover:border-accent/30 hover:text-foreground"
                >
                  <Github className="h-4 w-4" />
                  GitHub
                </Link>
                <Link
                  href={site.links.linkedin}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-2 rounded-full border border-line px-4 py-3 text-sm text-muted hover:border-accent/30 hover:text-foreground"
                >
                  <Linkedin className="h-4 w-4" />
                  LinkedIn
                </Link>
                <Link
                  href="/resume.pdf"
                  className="inline-flex items-center gap-2 rounded-full border border-accent/30 bg-accentSoft px-4 py-3 text-sm text-foreground hover:border-accent/50"
                >
                  View CV
                </Link>
              </div>
            </div>
          </div>
        </div>
      </Container>
    </Reveal>
  );
}
