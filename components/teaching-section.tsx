import { teaching, activities } from "@/data/teaching";
import { Container } from "@/components/ui/container";
import { Reveal } from "@/components/ui/reveal";
import { SectionHeader } from "@/components/section-header";

export function TeachingSection() {
  return (
    <Reveal>
      <Container className="px-6 py-8 sm:px-8 sm:py-10 lg:px-10">
        <div className="space-y-8">
          <SectionHeader
            label="/ teaching"
            title="Teaching"
            description="Teaching work I've done alongside research."
          />

          <div className="grid gap-4">
            {teaching.map((item) => (
              <article key={`${item.title}-${item.period}`} className="rounded-[28px] border border-line bg-white/[0.03] p-6">
                <div className="flex flex-wrap items-start justify-between gap-4">
                  <div>
                    <h3 className="text-xl font-semibold text-foreground">{item.title}</h3>
                    <p className="mt-2 text-sm text-muted">{item.organization}</p>
                  </div>
                  <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-accent">{item.period}</p>
                </div>

                <p className="mt-4 text-sm leading-7 text-slate-200">{item.summary}</p>

                <div className="mt-4 flex flex-wrap gap-2">
                  {item.courses.map((course) => (
                    <span
                      key={course}
                      className="rounded-full border border-line bg-white/[0.03] px-3 py-1.5 text-sm text-slate-200"
                    >
                      {course}
                    </span>
                  ))}
                </div>
              </article>
            ))}
          </div>

          <div className="rounded-[28px] border border-line bg-white/[0.03] p-6">
            <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-accent">Activities</p>
            <ul className="mt-4 space-y-3">
              {activities.map((item) => (
                <li key={item.role} className="flex gap-3 text-sm">
                  <span className="mt-[0.4rem] h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />
                  <span>
                    <span className="font-medium text-slate-200">{item.role}</span>
                    <span className="text-muted"> — {item.organization}</span>
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </Container>
    </Reveal>
  );
}
