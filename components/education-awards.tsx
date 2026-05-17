import { awards, education } from "@/data/educationAwards";
import { Container } from "@/components/ui/container";
import { Reveal } from "@/components/ui/reveal";
import { SectionHeader } from "@/components/section-header";

export function EducationAwards() {
  return (
    <Reveal>
      <Container id="education" className="px-6 py-8 sm:px-8 sm:py-10 lg:px-10">
        <div className="space-y-8">
          <SectionHeader
            label="/ education"
            title="Education and awards"
            description="Academic training, research preparation, and recent recognitions."
          />

          <div className="grid gap-5 lg:grid-cols-[1.2fr_0.8fr]">
            <div className="space-y-4">
              {education.map((item) => (
                <article key={`${item.institution}-${item.credential}`} className="rounded-[28px] border border-line bg-white/[0.03] p-6">
                  <div className="flex flex-wrap items-start justify-between gap-4">
                    <div>
                      <h3 className="text-xl font-semibold text-foreground">{item.credential}</h3>
                      <p className="mt-2 text-sm text-muted">
                        {item.institution}
                        {item.location ? ` - ${item.location}` : ""}
                      </p>
                    </div>
                    <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-accent">{item.period}</p>
                  </div>

                  {item.details.length > 0 && (
                    <ul className="mt-4 space-y-2 text-sm leading-7 text-slate-300">
                      {item.details.map((detail) => (
                        <li key={detail} className="flex gap-3">
                          <span className="mt-3 h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />
                          <span>{detail}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                </article>
              ))}
            </div>

            <div className="rounded-[28px] border border-accent/20 bg-accentSoft p-6">
              <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-accent">Awards and honors</p>
              <div className="mt-5 space-y-5">
                {awards.map((award) => (
                  <article key={award.title} className="border-b border-line pb-5 last:border-b-0 last:pb-0">
                    <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted">{award.year}</p>
                    <h3 className="mt-2 text-lg font-semibold leading-7 text-foreground">{award.title}</h3>
                    <p className="mt-2 text-sm leading-7 text-muted">{award.organization}</p>
                  </article>
                ))}
              </div>
            </div>
          </div>
        </div>
      </Container>
    </Reveal>
  );
}
