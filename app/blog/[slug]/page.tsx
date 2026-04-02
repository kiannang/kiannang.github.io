import Image from "next/image";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { getAllPosts, getPostBySlug } from "@/lib/posts";
import { Navbar } from "@/components/navbar";
import { Footer } from "@/components/footer";
import { ShareButton } from "@/components/share-button";

export async function generateStaticParams() {
  return getAllPosts().map((post) => ({ slug: post.slug }));
}

export default async function PostPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const post = await getPostBySlug(slug);

  return (
    <div className="relative min-h-screen">
      <Navbar />

      <main className="mx-auto w-full max-w-3xl px-4 pb-24 pt-32 sm:px-6 lg:px-8">
        <div className="space-y-10">

          {/* Back link */}
          <Link
            href="/blog"
            className="inline-flex items-center gap-2 font-mono text-xs uppercase tracking-[0.22em] text-muted hover:text-foreground"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            All posts
          </Link>

          {/* 1. Title + meta */}
          <div className="space-y-4">
            <div className="flex flex-wrap gap-2">
              {post.tags.map((tag) => (
                <span
                  key={tag}
                  className="rounded-full border border-line px-3 py-0.5 font-mono text-[11px] uppercase tracking-[0.2em] text-muted"
                >
                  {tag}
                </span>
              ))}
            </div>
            <h1 className="font-display italic text-5xl leading-tight text-foreground sm:text-6xl">
              {post.title}
            </h1>
            <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted">
              {new Date(post.date).toLocaleDateString("en-US", {
                year: "numeric",
                month: "long",
                day: "numeric",
              })}
            </p>
          </div>

          {/* 2. Featured landscape image */}
          {post.image && (
            <div className="space-y-2">
              <div className="relative aspect-[16/9] w-full overflow-hidden rounded-[28px] border border-line">
                <Image
                  src={post.image}
                  alt={post.title}
                  fill
                  className="object-cover"
                  sizes="(max-width: 768px) 100vw, 768px"
                />
              </div>
              {post.imageCredit && (
                <p className="text-right font-mono text-[11px] text-muted">
                  Credit: {post.imageCredit}
                </p>
              )}
            </div>
          )}

          {/* 3–8. Post body (intro, headings, content, conclusion) */}
          <div
            className="prose-content"
            dangerouslySetInnerHTML={{ __html: post.contentHtml }}
          />

          {/* 9. Share */}
          <div className="border-t border-line pt-8 space-y-3">
            <p className="font-mono text-xs uppercase tracking-[0.22em] text-muted">Share this post</p>
            <ShareButton title={post.title} />
          </div>

        </div>
      </main>

      <Footer />
    </div>
  );
}
