"use client";

import Link from "next/link";
import { ArrowUpRight, ChevronDown } from "lucide-react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useState } from "react";
import { selectedWork } from "@/data/selectedWork";
import { Container } from "@/components/ui/container";
import { Reveal } from "@/components/ui/reveal";
import { SectionHeader } from "@/components/section-header";
import { cn } from "@/lib/utils";

export function SelectedWork() {
  const [openId, setOpenId] = useState<string | null>(selectedWork[0]?.id ?? null);
  const reduceMotion = useReducedMotion();

  return (
    <Reveal>
      <Container id="work" className="px-6 py-8 sm:px-8 sm:py-10 lg:px-10">
        <div className="space-y-8">
          <SectionHeader
            label="/ selected public work"
            title="Public-facing work, presented with restraint."
            description="This section focuses on work that is appropriate to share publicly: selected research contributions, technical systems work, and engineering roles that reflect how I think and build."
          />

          <div className="grid gap-4 xl:grid-cols-2">
            {selectedWork.map((item) => {
              const isOpen = openId === item.id;

              return (
                <article
                  key={item.id}
                  className={cn(
                    "rounded-[28px] border p-6 transition-all",
                    isOpen ? "border-accent/30 bg-white/[0.05]" : "border-line bg-white/[0.03]",
                  )}
                >
                  <div className="flex flex-wrap items-start justify-between gap-4">
                    <div className="space-y-4">
                      <div className="flex flex-wrap items-center gap-3">
                        <span className="rounded-full border border-accent/20 bg-accentSoft px-3 py-1 font-mono text-[11px] uppercase tracking-[0.22em] text-accent">
                          {item.category}
                        </span>
                        <span className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted">{item.role}</span>
                      </div>
                      <div>
                        <h3 className="text-2xl font-semibold text-foreground">{item.title}</h3>
                        <p className="mt-3 max-w-xl text-sm leading-7 text-muted">{item.summary}</p>
                      </div>
                    </div>

                    <button
                      type="button"
                      aria-expanded={isOpen}
                      onClick={() => setOpenId((current) => (current === item.id ? null : item.id))}
                      className="inline-flex items-center gap-2 rounded-full border border-line px-4 py-2 text-sm text-muted hover:border-accent/30 hover:text-foreground"
                    >
                      Details
                      <ChevronDown className={cn("h-4 w-4 transition-transform", isOpen && "rotate-180")} />
                    </button>
                  </div>

                  <AnimatePresence initial={false}>
                    {isOpen && (
                      <motion.div
                        initial={reduceMotion ? false : { height: 0, opacity: 0 }}
                        animate={reduceMotion ? {} : { height: "auto", opacity: 1 }}
                        exit={reduceMotion ? {} : { height: 0, opacity: 0 }}
                        transition={{ duration: 0.25 }}
                        className="overflow-hidden"
                      >
                        <div className="mt-6 grid gap-5 border-t border-line pt-6 sm:grid-cols-[1fr_auto]">
                          <div className="space-y-4">
                            <div>
                              <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted">Context</p>
                              <p className="mt-2 text-sm leading-7 text-slate-300">{item.details}</p>
                            </div>
                            <div className="flex flex-wrap gap-2">
                              {item.tools.map((tool) => (
                                <span
                                  key={tool}
                                  className="rounded-full border border-line bg-white/[0.03] px-2.5 py-1 font-mono text-[11px] uppercase tracking-[0.16em] text-slate-300"
                                >
                                  {tool}
                                </span>
                              ))}
                            </div>
                          </div>

                          {item.cta && (
                            <div className="flex items-start">
                              <Link
                                href={item.cta.href}
                                target={item.cta.external ? "_blank" : undefined}
                                rel={item.cta.external ? "noreferrer" : undefined}
                                className="inline-flex items-center gap-2 rounded-full border border-accent/30 bg-accentSoft px-4 py-2 text-sm text-foreground hover:border-accent/50"
                              >
                                {item.cta.label}
                                <ArrowUpRight className="h-4 w-4" />
                              </Link>
                            </div>
                          )}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </article>
              );
            })}
          </div>
        </div>
      </Container>
    </Reveal>
  );
}
