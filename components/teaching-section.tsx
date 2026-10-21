import { teaching, activities } from "@/data/teaching";
import { Container } from "@/components/ui/container";
import { Panel } from "@/components/ui/panel";
import { SectionHeader } from "@/components/section-header";

export function TeachingSection() {
  return (
    <Container id="teaching">
      <SectionHeader title="Teaching." />

      <div className="grid gap-4 md:grid-cols-[1.4fr_1fr]">
        <div className="space-y-4">
          {teaching.map((item) => (
            <Panel key={`${item.title}-${item.period}`} title={item.organization} aside={item.period}>
              <h3 className="font-display text-[22px] leading-tight tracking-[-0.02em] text-foreground">{item.title}</h3>
              <p className="mt-2 text-[13px] leading-6">{item.summary}</p>

              <div className="mt-3 flex flex-wrap gap-1.5">
                {item.courses.map((course) => (
                  <span key={course} className="rounded-full border border-line bg-accentSoft px-2.5 py-[1px] text-[11px]">
                    {course}
                  </span>
                ))}
              </div>
            </Panel>
          ))}
        </div>

        <Panel id="activities" title="Activities" className="self-start">
          <ul className="space-y-2 text-[13px] leading-5">
            {activities.map((item) => (
              <li key={item.role} className="flex gap-2">
                <span className="text-accent">&rsaquo;</span>
                <span>
                  <span className="font-bold">{item.role}</span>
                  <span className="text-muted"> — {item.organization}</span>
                </span>
              </li>
            ))}
          </ul>
        </Panel>
      </div>
    </Container>
  );
}
