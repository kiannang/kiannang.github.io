import { researchAreas } from "@/data/researchAreas";
import { Container } from "@/components/ui/container";
import { Reveal } from "@/components/ui/reveal";
import { SectionHeader } from "@/components/section-header";

export function ResearchAreas() {
  return (
    <Reveal>
      <Container className="px-6 py-8 sm:px-8 sm:py-10 lg:px-10">
        <div className="space-y-8">
          <SectionHeader
            label="/ research areas"
            title="Themes that shape the work."
            description="Rather than centering confidential project details, this portfolio highlights the research directions, methods, and systems questions that organize my work."
          />

          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            {researchAreas.map((area) => (
              <article
                key={area.title}
                className="group flex h-full flex-col rounded-[26px] border border-line bg-white/[0.03] p-5 hover:-translate-y-1 hover:border-accent/30 hover:bg-white/[0.05]"
              >
                <div className="mb-5 h-10 w-10 rounded-2xl border border-accent/20 bg-accentSoft text-accent" />
                <div className="space-y-3">
                  <h3 className="text-lg font-semibold text-foreground">{area.title}</h3>
                  <p className="text-sm leading-7 text-muted">{area.description}</p>
                </div>
                <div className="mt-5 flex flex-wrap gap-2">
                  {area.chips.map((chip) => (
                    <span
                      key={chip}
                      className="rounded-full border border-line bg-white/[0.03] px-2.5 py-1 font-mono text-[11px] uppercase tracking-[0.18em] text-slate-300"
                    >
                      {chip}
                    </span>
                  ))}
                </div>
              </article>
            ))}
          </div>
        </div>
      </Container>
    </Reveal>
  );
}
