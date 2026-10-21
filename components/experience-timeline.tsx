import { experience } from "@/data/experience";
import { Container } from "@/components/ui/container";
import { Panel } from "@/components/ui/panel";
import { SectionHeader } from "@/components/section-header";

export function ExperienceTimeline() {
  return (
    <Container id="experience">
      <SectionHeader title="Experience." />

      <div className="space-y-4">
        {experience.map((item) => (
          <Panel key={`${item.organization}-${item.title}`} title={item.organization} aside={item.period}>
            <h3 className="font-display text-[22px] leading-tight tracking-[-0.02em] text-foreground">{item.title}</h3>
            {item.summary && <p className="mt-1 text-[12px] leading-5 text-muted">{item.summary}</p>}

            {item.bullets.length > 0 && (
              <ul className="mt-3 space-y-1 text-[13px] leading-6">
                {item.bullets.map((bullet) => (
                  <li key={bullet} className="flex gap-2">
                    <span className="text-accent">&rsaquo;</span>
                    <span>{bullet}</span>
                  </li>
                ))}
              </ul>
            )}

            {item.tags.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-1.5">
                {item.tags.map((tag) => (
                  <span key={tag} className="rounded-full border border-line bg-accentSoft px-2.5 py-[1px] text-[11px]">
                    {tag}
                  </span>
                ))}
              </div>
            )}
          </Panel>
        ))}
      </div>
    </Container>
  );
}
