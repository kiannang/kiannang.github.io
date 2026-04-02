import { focusIntro, focusPanels } from "@/data/focus";
import { Container } from "@/components/ui/container";
import { Reveal } from "@/components/ui/reveal";
import { SectionHeader } from "@/components/section-header";

export function CurrentFocus() {
  return (
    <Reveal>
      <Container id="focus" className="px-6 py-8 sm:px-8 sm:py-10 lg:px-10">
        <div className="space-y-8">
          <SectionHeader
            label="/ current focus"
            title="Current work, shared at the right level."
            description={focusIntro}
          />

          <div className="grid gap-4 lg:grid-cols-[1.2fr_0.8fr_0.8fr]">
            <div className="rounded-[28px] border border-accent/20 bg-accentSoft p-6">
              <p className="font-mono text-[11px] uppercase tracking-[0.26em] text-accent">Overview</p>
              <p className="mt-4 max-w-xl text-lg leading-8 text-slate-100">
                I&apos;m currently working on research related to multimodal learning, perception, and human-centered AI.
                Some of that work is still in progress or under review, so I keep this section fairly high level.
              </p>
              <p className="mt-4 text-sm leading-7 text-slate-300">I&apos;m happy to share more context when appropriate.</p>
            </div>

            {focusPanels.slice(0, 2).map((panel) => (
              <div key={panel.title} className="rounded-[28px] border border-line bg-white/[0.03] p-6">
                <p className="font-mono text-[11px] uppercase tracking-[0.24em] text-accent">{panel.label}</p>
                <h3 className="mt-4 text-xl font-semibold text-foreground">{panel.title}</h3>
                <p className="mt-3 text-sm leading-7 text-muted">{panel.description}</p>
              </div>
            ))}
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            {focusPanels.slice(2).map((panel) => (
              <div key={panel.title} className="rounded-[28px] border border-line bg-white/[0.03] p-6">
                <p className="font-mono text-[11px] uppercase tracking-[0.24em] text-accent">{panel.label}</p>
                <h3 className="mt-4 text-xl font-semibold text-foreground">{panel.title}</h3>
                <p className="mt-3 text-sm leading-7 text-muted">{panel.description}</p>
              </div>
            ))}
          </div>
        </div>
      </Container>
    </Reveal>
  );
}
