"use client";

import Link from "next/link";
import { ArrowRight, Download, MapPin } from "lucide-react";
import { motion, useReducedMotion } from "framer-motion";
import { site } from "@/data/site";
import { Container } from "@/components/ui/container";

export function Hero() {
  const reduceMotion = useReducedMotion();

  return (
    <Container id="home" className="px-6 py-8 sm:px-8 sm:py-10 lg:px-12 lg:py-14">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_18%,rgba(124,199,255,0.14),transparent_24%),radial-gradient(circle_at_82%_30%,rgba(126,120,255,0.12),transparent_20%)]" />
      <div className="absolute right-10 top-10 hidden h-20 w-20 rounded-full border border-accent/20 bg-accent/10 blur-2xl lg:block" />

      <div className="relative grid gap-10 lg:grid-cols-[1.2fr_0.8fr] lg:items-center">
        <motion.div
          initial={reduceMotion ? false : { opacity: 0, y: 18 }}
          animate={reduceMotion ? {} : { opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
          className="space-y-8"
        >
          <div className="space-y-5">
            <div className="inline-flex items-center gap-3 rounded-full border border-accent/20 bg-accentSoft px-4 py-2 font-mono text-[11px] uppercase tracking-[0.28em] text-accent">
              <span className="h-2 w-2 rounded-full bg-accent" />
              Open to collaboration
            </div>
            <div className="space-y-5">
              <p className="font-mono text-xs uppercase tracking-[0.32em] text-muted">Researcher / Engineer</p>
              <h1 className="max-w-4xl text-4xl font-semibold tracking-tight text-foreground sm:text-5xl lg:text-6xl">
                Kianna Ng
              </h1>
              <div className="space-y-3 text-base text-muted sm:text-lg">
                <p className="max-w-2xl text-xl font-medium leading-8 text-slate-100 sm:text-2xl sm:leading-9">
                  PhD Student in Electrical Engineering and Computer Science at the University of California, Merced.
                </p>
                <p className="max-w-2xl text-lg leading-8 text-slate-300 sm:text-xl">
                  I build human-centered AI systems for perception, safety, and intelligent interaction.
                </p>
                <p className="max-w-2xl leading-8">
                  My work spans multimodal learning, robotics, sensing systems, autonomous systems, and trustworthy AI,
                  with an emphasis on how technical systems meet human needs in real environments.
                </p>
              </div>
            </div>
          </div>

          <div className="flex flex-wrap gap-3">
            <Link
              href="#work"
              className="inline-flex items-center gap-2 rounded-full border border-accent/30 bg-accent px-5 py-3 text-sm font-medium text-slate-950 hover:translate-y-[-1px] hover:bg-[#95d5ff]"
            >
              View Work
              <ArrowRight className="h-4 w-4" />
            </Link>
            <Link
              href="/resume.pdf"
              className="inline-flex items-center gap-2 rounded-full border border-line bg-white/5 px-5 py-3 text-sm font-medium text-foreground hover:translate-y-[-1px] hover:border-accent/30 hover:bg-accentSoft"
            >
              <Download className="h-4 w-4" />
              Download CV
            </Link>
            <Link
              href="#contact"
              className="inline-flex items-center gap-2 rounded-full border border-line bg-transparent px-5 py-3 text-sm font-medium text-muted hover:translate-y-[-1px] hover:border-white/20 hover:text-foreground"
            >
              Contact Me
            </Link>
          </div>
        </motion.div>

        <motion.aside
          initial={reduceMotion ? false : { opacity: 0, x: 18 }}
          animate={reduceMotion ? {} : { opacity: 1, x: 0 }}
          transition={{ duration: 0.7, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
          className="relative overflow-hidden rounded-[28px] border border-white/10 bg-panelStrong p-6 shadow-glow"
        >
          <div className="absolute inset-0 bg-[linear-gradient(140deg,rgba(124,199,255,0.08),transparent_40%,rgba(126,120,255,0.08))]" />
          <div className="relative space-y-6">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="font-mono text-[11px] uppercase tracking-[0.28em] text-accent">Profile</p>
                <h2 className="mt-3 text-2xl font-semibold text-foreground">Kianna Ng</h2>
                <p className="mt-2 max-w-xs text-sm leading-6 text-muted">{site.role}</p>
              </div>
              <div className="rounded-full border border-accent/20 bg-accentSoft px-3 py-1 font-mono text-[11px] uppercase tracking-[0.24em] text-accent">
                UC Merced
              </div>
            </div>

            <div className="grid gap-3 rounded-[24px] border border-line bg-white/[0.02] p-4">
              <div>
                <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted">Affiliation</p>
                <p className="mt-1 text-sm text-slate-200">{site.affiliation}</p>
              </div>
              <div className="flex items-center gap-2 text-sm text-muted">
                <MapPin className="h-4 w-4 text-accent" />
                {site.location}
              </div>
            </div>

            <div className="space-y-3">
              <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted">Research Tags</p>
              <div className="flex flex-wrap gap-2">
                {site.heroTags.map((tag) => (
                  <span
                    key={tag}
                    className="rounded-full border border-line bg-white/[0.03] px-3 py-1.5 text-sm text-slate-200"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <div className="rounded-[22px] border border-line bg-white/[0.03] p-4">
                <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted">Current Lens</p>
                <p className="mt-2 text-sm leading-6 text-slate-300">
                  Multimodal perception, intelligent systems, and careful human-machine interaction design.
                </p>
              </div>
              <div className="rounded-[22px] border border-line bg-white/[0.03] p-4">
                <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted">Approach</p>
                <p className="mt-2 text-sm leading-6 text-slate-300">
                  Research-forward implementation with attention to sensing, safety, and real-world workflow constraints.
                </p>
              </div>
            </div>
          </div>
        </motion.aside>
      </div>
    </Container>
  );
}
