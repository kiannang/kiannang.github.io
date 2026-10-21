import { awards, education } from "@/data/educationAwards";
import { Container } from "@/components/ui/container";
import { Panel } from "@/components/ui/panel";
import { SectionHeader } from "@/components/section-header";

export function EducationAwards() {
  return (
    <Container id="education">
      <SectionHeader title="Education." />

      <div className="grid gap-4 md:grid-cols-[1.4fr_1fr]">
        <div className="space-y-4">
          {education.map((item) => (
            <Panel key={`${item.institution}-${item.credential}`} title={item.institution} aside={item.period}>
              <h3 className="font-display text-[22px] leading-tight tracking-[-0.02em] text-foreground">{item.credential}</h3>
              {item.location && <p className="mt-1 text-[12px] text-muted">{item.location}</p>}

              {item.details.length > 0 && (
                <ul className="mt-3 space-y-1 text-[13px] leading-6">
                  {item.details.map((detail) => (
                    <li key={detail} className="flex gap-2">
                      <span className="text-accent">&rsaquo;</span>
                      <span>{detail}</span>
                    </li>
                  ))}
                </ul>
              )}
            </Panel>
          ))}
        </div>

        <Panel id="awards" title="Awards and Honors" className="self-start">
          <div className="space-y-4">
            {awards.map((award) => (
              <article key={award.title} className="border-b border-dotted border-line pb-4 last:border-b-0 last:pb-0">
                <p className="text-[11px] font-bold text-muted">{award.year}</p>
                <h3 className="mt-1 text-[13px] font-bold leading-5 text-foreground">{award.title}</h3>
                <p className="mt-1 text-[12px] leading-5 text-muted">{award.organization}</p>
              </article>
            ))}
          </div>
        </Panel>
      </div>
    </Container>
  );
}
