"use client";

import Image from "next/image";
import Link from "next/link";
import { ArrowRight, Download } from "lucide-react";
import { motion, useReducedMotion } from "framer-motion";
import { Container } from "@/components/ui/container";

export function Hero() {
  const reduceMotion = useReducedMotion();

  return (
    <Container id="home" className="px-6 py-10 sm:px-8 sm:py-14 lg:px-12 lg:py-20">
      <div className="grid gap-12 lg:grid-cols-[0.75fr_1.25fr] lg:items-center">

        {/* Photo */}
        <motion.div
          initial={reduceMotion ? false : { opacity: 0, x: -20 }}
          animate={reduceMotion ? {} : { opacity: 1, x: 0 }}
          transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
        >
          <div className="relative aspect-[3/4] overflow-hidden rounded-[40px] border border-accent/20">
            <Image
              src="/images/kianna_headshot.png"
              alt="Kianna Ng"
              fill
              className="object-cover object-top"
              sizes="(max-width: 1024px) 100vw, 35vw"
              priority
            />
          </div>
        </motion.div>

        {/* Text */}
        <motion.div
          initial={reduceMotion ? false : { opacity: 0, y: 20 }}
          animate={reduceMotion ? {} : { opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
          className="space-y-8"
        >
          <div className="inline-flex items-center gap-3 rounded-full border border-accent/20 bg-accentSoft px-4 py-2 font-mono text-[11px] uppercase tracking-[0.28em] text-accent">
            <span className="h-2 w-2 rounded-full bg-accent" />
            Open to collaboration
          </div>

          <div className="space-y-1">
            <p className="font-mono text-xs uppercase tracking-[0.32em] text-muted">
              PhD Student · UC Merced
            </p>
            <h1 className="leading-none">
              <span className="block text-5xl font-black text-foreground sm:text-6xl lg:text-7xl">
                Hi, I&apos;m
              </span>
              <span className="block font-display text-6xl text-accent sm:text-7xl lg:text-8xl">
                Kianna Ng.
              </span>
            </h1>
          </div>

          <div className="space-y-3 max-w-xl">
            <p className="text-xl font-medium leading-8 text-foreground sm:text-2xl">
              I build computer vision and perception systems for autonomous driving safety and human-centered AI.
            </p>
            <p className="text-base leading-8 text-muted">
              My work spans robotic perception, multitask representation learning, multimodal sensing, and intelligent systems.
            </p>
          </div>

          <div className="flex flex-wrap gap-3">
            <Link
              href="#publications"
              className="inline-flex items-center gap-2 rounded-full bg-accent px-6 py-3 text-sm font-semibold text-[#3e2723] hover:bg-accent/80"
            >
              View Work
              <ArrowRight className="h-4 w-4" />
            </Link>
            <Link
              href="/resume/CV.pdf"
              className="inline-flex items-center gap-2 rounded-full border border-accent/30 bg-accentSoft px-6 py-3 text-sm font-medium text-foreground hover:border-accent/60"
            >
              <Download className="h-4 w-4" />
              Download CV
            </Link>
            <Link
              href="#contact"
              className="inline-flex items-center gap-2 rounded-full border border-line px-6 py-3 text-sm font-medium text-muted hover:border-accent/30 hover:text-foreground"
            >
              Contact Me
            </Link>
          </div>
        </motion.div>

      </div>
    </Container>
  );
}
