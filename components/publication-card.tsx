import Link from "next/link";
import { ArrowUpRight } from "lucide-react";
import { publications } from "@/data/publications";
import { Container } from "@/components/ui/container";
import { Reveal } from "@/components/ui/reveal";
import { SectionHeader } from "@/components/section-header";

export function PublicationCard() {
  const publication = publications[0];

  return (
    <Reveal>
      <Container id="publications" className="px-6 py-8 sm:px-8 sm:py-10 lg:px-10">
        <div className="space-y-8">
          <SectionHeader
            label="/ publications"
            title="Recent published work."
            description="Peer-reviewed research in multimodal learning and intelligent systems."
          />

          <article className="rounded-[30px] border border-accent/20 bg-[linear-gradient(145deg,rgba(124,199,255,0.1),rgba(255,255,255,0.03)_38%,rgba(126,120,255,0.08))] p-6 sm:p-8">
            <div className="flex flex-wrap items-center gap-3">
              <span className="rounded-full border border-accent/20 bg-accentSoft px-3 py-1 font-mono text-[11px] uppercase tracking-[0.22em] text-accent">
                Featured publication
              </span>
              <span className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted">{publication.venue}</span>
              <span className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted">{publication.year}</span>
            </div>

            <div className="mt-6 grid gap-8 lg:grid-cols-[1.2fr_0.8fr]">
              <div className="space-y-5">
                <h3 className="max-w-3xl text-2xl font-semibold leading-tight text-foreground sm:text-3xl">
                  {publication.title}
                </h3>
                <p className="text-sm leading-7 text-muted">{publication.authors}</p>
                <p className="max-w-3xl text-base leading-8 text-slate-200">{publication.summary}</p>
              </div>

              <div className="rounded-[26px] border border-line bg-panelStrong p-5">
                <div className="space-y-4">
                  <div>
                    <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted">Venue</p>
                    <p className="mt-2 text-sm text-slate-200">{publication.fullVenue}</p>
                  </div>
                  <div>
                    <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted">Citation note</p>
                    <p className="mt-2 text-sm leading-7 text-muted">Available via DOI and institutional repository.</p>
                  </div>
                  <div className="flex flex-wrap gap-3 pt-2">
                    <Link
                      href={publication.links.paper}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-2 rounded-full border border-accent/30 bg-accentSoft px-4 py-2 text-sm text-foreground hover:border-accent/50"
                    >
                      View Paper
                      <ArrowUpRight className="h-4 w-4" />
                    </Link>
                    <Link
                      href={publication.links.doi}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-2 rounded-full border border-line px-4 py-2 text-sm text-muted hover:border-accent/30 hover:text-foreground"
                    >
                      DOI
                      <ArrowUpRight className="h-4 w-4" />
                    </Link>
                    <span className="inline-flex items-center rounded-full border border-line px-4 py-2 text-sm text-muted">
                      Citation available on request
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </article>


        </div>
      </Container>
    </Reveal>
  );
}
