import { experience } from "@/data/experience";
import { Container } from "@/components/ui/container";
import { Reveal } from "@/components/ui/reveal";
import { SectionHeader } from "@/components/section-header";

export function ExperienceTimeline() {
  return (
    <Reveal>
      <Container id="experience" className="px-6 py-8 sm:px-8 sm:py-10 lg:px-10">
        <div className="space-y-8">
          <SectionHeader
            label="/ experience"
            title="Research and engineering experience across labs and systems."
            description="A mix of research, engineering, and hands-on systems work across labs and academic projects."
          />

          <div className="relative space-y-5 before:absolute before:left-4 before:top-2 before:hidden before:h-[calc(100%-1rem)] before:w-px before:bg-line lg:before:block">
            {experience.map((item) => (
              <article
                key={`${item.organization}-${item.title}`}
                className="relative rounded-[28px] border border-line bg-white/[0.03] p-6 lg:ml-10 lg:grid lg:grid-cols-[220px_1fr] lg:gap-8"
              >
                <div className="relative">
                  <span className="absolute -left-[3.15rem] top-2 hidden h-3 w-3 rounded-full border border-accent/30 bg-accent lg:block" />
                  <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-accent">{item.period}</p>
                  <p className="mt-2 text-sm text-muted">{item.organization}</p>
                </div>

                <div className="mt-5 space-y-4 lg:mt-0">
                  <div>
                    <h3 className="text-xl font-semibold text-foreground">{item.title}</h3>
                    <p className="mt-2 text-sm leading-7 text-muted">{item.summary}</p>
                  </div>

                  <ul className="space-y-3 text-sm leading-7 text-slate-300">
                    {item.bullets.map((bullet) => (
                      <li key={bullet} className="flex gap-3">
                        <span className="mt-3 h-1.5 w-1.5 rounded-full bg-accent" />
                        <span>{bullet}</span>
                      </li>
                    ))}
                  </ul>

                  <div className="flex flex-wrap gap-2 pt-1">
                    {item.tags.map((tag) => (
                      <span
                        key={tag}
                        className="rounded-full border border-line bg-white/[0.03] px-2.5 py-1 font-mono text-[11px] uppercase tracking-[0.16em] text-slate-300"
                      >
                        {tag}
                      </span>
                    ))}
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
