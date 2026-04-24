"use client";

import Link from "next/link";
import { ArrowUpRight } from "lucide-react";
import { Container } from "@/components/ui/container";
import { Reveal } from "@/components/ui/reveal";
import { SectionHeader } from "@/components/section-header";
import { funProjects } from "@/data/funProjects";

export function FunProjects() {
  return (
    <Reveal>
      <Container id="projects" className="px-6 py-8 sm:px-8 sm:py-10 lg:px-10">
        <div className="space-y-8">
          <SectionHeader
            label="/ fun projects"
            title="A lighter side of the work."
            description="A small space for projects that are more playful, experimental, or just fun to share."
          />

          <div className="grid gap-4">
            {funProjects.map((project) => (
              <article
                key={project.id}
                className="rounded-[28px] border border-line bg-white/[0.03] p-6"
              >
                <div className="flex flex-col gap-6 sm:flex-row sm:items-start sm:justify-between">
                  <div className="max-w-2xl space-y-4">
                    <span className="inline-flex rounded-full border border-accent/20 bg-accentSoft px-3 py-1 font-mono text-[11px] uppercase tracking-[0.22em] text-accent">
                      {project.category}
                    </span>
                    <div>
                      <h3 className="text-2xl font-semibold text-foreground">{project.title}</h3>
                      <p className="mt-3 text-sm leading-7 text-muted">{project.summary}</p>
                      <p className="mt-3 text-sm leading-7 text-slate-300">{project.details}</p>
                    </div>
                  </div>

                  <div className="flex items-start">
                    <Link
                      href={project.href}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-2 rounded-full border border-accent/30 bg-accentSoft px-4 py-2 text-sm text-foreground hover:border-accent/50"
                    >
                      {project.ctaLabel}
                      <ArrowUpRight className="h-4 w-4" />
                    </Link>
                  </div>
                </div>
              </article>
            ))}
          </div>
        </div>
      </Container>
    </Reveal>
  );
}
