/* M3 docs — page polish JS.

   Two responsibilities:

   1. Scroll-triggered fade-up:
      - Content is visible by default (CSS).
      - JS finds animation candidates, tags below-fold ones with
        `.anim-pending` (which sets opacity:0 + translate via CSS),
        and uses IntersectionObserver to flip them to `.is-visible`
        as they enter the viewport.
      - If anything fails (JS disabled, iframe with innerHeight=0,
        broken observer), no element ever gets `.anim-pending` →
        content remains fully visible. Fail-open by design.

   2. Sticky header shadow on scroll. */

(function () {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  const ANIM_SELECTOR =
    '.md-typeset .grid.cards > ul > li, ' +
    '.md-typeset .grid:not(.cards) > *, ' +
    '.anim-fade-up';

  const initFadeUp = () => {
    const targets = document.querySelectorAll(ANIM_SELECTOR);
    if (!targets.length) return;
    if (!('IntersectionObserver' in window)) return;

    const vh = window.innerHeight || document.documentElement.clientHeight || 0;
    // If viewport is degenerate (0 or unknown), bail — keep content visible.
    if (vh < 100) return;

    // Tag below-the-fold targets as pending (CSS hides them); keep above-fold
    // content fully visible to avoid any flash of empty space.
    targets.forEach(el => {
      const r = el.getBoundingClientRect();
      // r.top here is relative to viewport. If element's top is more than ~50px
      // below the visible area, animate. Otherwise leave visible.
      if (r.top > vh - 40) {
        el.classList.add('anim-pending');
      }
    });

    const io = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          io.unobserve(entry.target);
        }
      });
    }, {
      rootMargin: '0px 0px 80px 0px',   // pre-reveal 80px before entering
      threshold: 0.01,
    });

    targets.forEach(el => {
      if (el.classList.contains('anim-pending')) io.observe(el);
    });
  };

  const stickyHeader = () => {
    const header = document.querySelector('.md-header');
    if (!header) return;
    const setShadow = () => header.setAttribute('data-md-state', window.scrollY > 8 ? 'shadow' : '');
    setShadow();                        // refresh state on every (re-)init
    if (window.__stickyWired) return;   // but attach the scroll listener only ONCE
    window.__stickyWired = true;        // (instant-nav re-runs init; don't stack handlers)
    let ticking = false;
    window.addEventListener('scroll', () => {
      if (!ticking) { requestAnimationFrame(() => { setShadow(); ticking = false; }); ticking = true; }
    }, { passive: true });
  };

  /* Scroll progress bar (Bioconductor reference) — slim violet bar at top */
  const scrollProgress = () => {
    if (document.querySelector('.scroll-progress')) return;
    const bar = document.createElement('div');
    bar.className = 'scroll-progress';
    document.body.appendChild(bar);

    let ticking = false;
    const update = () => {
      const h = document.documentElement;
      const total = h.scrollHeight - h.clientHeight;
      const pct = total > 0 ? (window.scrollY / total) * 100 : 0;
      bar.style.width = pct + '%';
      bar.classList.toggle('is-scrolling', window.scrollY > 80);
      ticking = false;
    };
    window.addEventListener('scroll', () => {
      if (!ticking) { requestAnimationFrame(update); ticking = true; }
    }, { passive: true });
    update();
  };

  /* Back-to-top button — fastai-style, appears after 600px scroll */
  const backToTop = () => {
    if (document.querySelector('.back-to-top')) return;
    const btn = document.createElement('button');
    btn.className = 'back-to-top';
    btn.setAttribute('aria-label', 'Back to top');
    btn.innerHTML = '↑';
    btn.addEventListener('click', () => {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
    document.body.appendChild(btn);

    let ticking = false;
    const update = () => {
      btn.classList.toggle('is-visible', window.scrollY > 600);
      ticking = false;
    };
    window.addEventListener('scroll', () => {
      if (!ticking) { requestAnimationFrame(update); ticking = true; }
    }, { passive: true });
    update();
  };

  const enableSectionNumbersOnHome = () => {
    const path = (window.location.pathname || '').replace(/\/+$/, '/');
    // A landing page is any page carrying the `.lp` wrapper (portal + every tool
    // home) — detect that so the hero styling is identical across all of them,
    // not just at the hardcoded `/m3/` path or the site root.
    const isHome = !!document.querySelector('.lp') || /\/m3\/$/.test(path) || path === '/' || /index\.html?$/i.test(path);
    if (isHome) document.body.classList.add('has-section-numbers');
    // Tutorial pages (notebooks) get step-numbering
    if (/\/notebooks\//.test(path)) {
      document.body.classList.add('has-tutorial-numbers');
    }
  };

  /* Map each H2 to an editorial eyebrow label for the section divider.
     The CSS ::before uses attr(data-eyebrow); fall-back is "Section". */
  const tagSectionEyebrows = () => {
    if (!document.body.classList.contains('has-section-numbers')) return;
    const map = {
      'how-it-works': 'How it works',
      'start-here':   'Start here',
      'why-m3':       'Why M3',
      'cite-m3':      'Citation',
    };
    document.querySelectorAll('.md-typeset h2[id]').forEach((h) => {
      const slug = h.id;
      if (map[slug]) h.setAttribute('data-eyebrow', map[slug]);
    });
  };

  /* Hero reveal sequence — Linear-style restrained orchestration.
     One reveal per browser session via sessionStorage gate.
     Sequence (total ~1200ms):
       t=0     hero eyebrow fades in
       t=100   wordmark word-by-word stagger
       t=600   tagline slides up
       t=750   CTA pair scales + fades in
       t=900   right column (illustration) fades in */
  const heroReveal = () => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    if (!document.body.classList.contains('has-section-numbers')) return; // home only
    if (sessionStorage.getItem('m3-hero-seen') === '1') return;
    return;  // fade-free hero: render the title instantly, skip the opacity-0 letter-stagger

    const eyebrow = document.querySelector('.hero-eyebrow');
    const h1      = document.querySelector('.md-typeset > h1:first-of-type');
    const tagline = document.querySelector('.hero-tagline');
    const cta     = document.querySelector('.hero-cta');
    const badges  = document.querySelector('.hero-badges');
    const illus   = document.querySelector('.hero-illustration');

    const stage = (el, css) => {
      if (!el) return;
      Object.assign(el.style, css);
    };

    // Stage: initial state (hidden offset)
    stage(eyebrow, { opacity: '0', transform: 'translateY(6px)', transition: 'opacity 400ms ease, transform 400ms ease' });
    stage(tagline, { opacity: '0', transform: 'translateY(8px)', transition: 'opacity 400ms ease, transform 400ms ease' });
    stage(cta,     { opacity: '0', transform: 'scale(0.97)',       transition: 'opacity 400ms ease, transform 400ms ease' });
    stage(badges,  { opacity: '0',                                  transition: 'opacity 400ms ease' });
    stage(illus,   { opacity: '0', transform: 'translateY(8px)', transition: 'opacity 500ms ease, transform 500ms ease' });

    // Wordmark: split into letters for stagger
    let wordmarkSpans = [];
    if (h1) {
      const text = h1.textContent.trim();
      h1.textContent = '';
      [...text].forEach((ch, i) => {
        const sp = document.createElement('span');
        sp.textContent = ch === ' ' ? ' ' : ch;
        sp.style.opacity = '0';
        sp.style.display = 'inline-block';
        sp.style.transform = 'translateY(12px)';
        sp.style.transition = 'opacity 380ms cubic-bezier(0.22,1,0.36,1), transform 380ms cubic-bezier(0.22,1,0.36,1)';
        sp.style.transitionDelay = (100 + i * 35) + 'ms';
        h1.appendChild(sp);
        wordmarkSpans.push(sp);
      });
    }

    // Reveal sequence
    requestAnimationFrame(() => {
      setTimeout(() => stage(eyebrow, { opacity: '1', transform: 'translateY(0)' }), 0);
      wordmarkSpans.forEach(sp => {
        sp.style.opacity = '1';
        sp.style.transform = 'translateY(0)';
      });
      setTimeout(() => stage(tagline, { opacity: '1', transform: 'translateY(0)' }), 600);
      setTimeout(() => stage(cta,     { opacity: '1', transform: 'scale(1)' }),    750);
      setTimeout(() => stage(badges,  { opacity: '1' }),                          800);
      setTimeout(() => stage(illus,   { opacity: '1', transform: 'translateY(0)' }), 900);
    });

    sessionStorage.setItem('m3-hero-seen', '1');
  };

  /* === SWISS-WHITE v0.5 ADDITIONS === */

  /* Magnetic buttons — opt-in via .magnetic class only.
     (Get Started / .lp-cta intentionally excluded — plain button, no pull.) */
  const magneticButtons = () => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    if (window.matchMedia('(pointer: coarse)').matches) return; // touch devices
    const targets = document.querySelectorAll('.magnetic');
    targets.forEach((el) => {
      let raf = null;
      const STRENGTH = 0.18;   // 0..1, higher = more pull
      const RANGE   = 90;      // px of activation radius beyond button bounds
      el.addEventListener('mousemove', (e) => {
        if (raf) cancelAnimationFrame(raf);
        raf = requestAnimationFrame(() => {
          const r = el.getBoundingClientRect();
          const cx = r.left + r.width / 2;
          const cy = r.top + r.height / 2;
          const dx = e.clientX - cx;
          const dy = e.clientY - cy;
          const dist = Math.hypot(dx, dy);
          const max = Math.max(r.width, r.height) / 2 + RANGE;
          if (dist < max) {
            el.style.transform = `translate3d(${dx * STRENGTH}px, ${dy * STRENGTH}px, 0)`;
          }
        });
      });
      el.addEventListener('mouseleave', () => {
        if (raf) cancelAnimationFrame(raf);
        el.style.transform = '';
      });
    });
  };

  /* Section scroll-spy rail — left-fixed §1 §2 §3 navigator */
  const sectionRail = () => {
    if (!document.body.classList.contains('has-section-numbers')) return;
    if (window.innerWidth < 1280) return;
    const sections = document.querySelectorAll('.lp-hero, .lp-section');
    if (sections.length === 0) return;

    /* Build the rail */
    const rail = document.createElement('aside');
    rail.className = 'section-rail';
    rail.setAttribute('aria-label', 'Section navigation');
    sections.forEach((s, i) => {
      const item = document.createElement('a');
      item.className = 'section-rail__item';
      item.setAttribute('data-rail-idx', String(i));
      item.textContent = '§' + (i + 1);
      item.addEventListener('click', (e) => {
        e.preventDefault();
        s.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
      rail.appendChild(item);
    });
    document.body.appendChild(rail);

    /* Activate on scroll via IntersectionObserver */
    const io = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const idx = Array.prototype.indexOf.call(sections, entry.target);
          rail.querySelectorAll('.section-rail__item').forEach((it, i) => {
            it.classList.toggle('is-active', i === idx);
          });
        }
      });
    }, { threshold: 0.3, rootMargin: '-30% 0px -30% 0px' });
    sections.forEach((s) => io.observe(s));
  };

  /* Number counter — count up to target on viewport entry */
  const numberCounters = () => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const els = document.querySelectorAll('.count-up');
    if (!els.length || !('IntersectionObserver' in window)) return;
    const io = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const el = entry.target;
        const target = parseFloat(el.getAttribute('data-target') || el.textContent || '0');
        const duration = 450;
        const start = performance.now();
        const easeOutQuart = (t) => 1 - Math.pow(1 - t, 4);
        const step = (now) => {
          const t = Math.min(1, (now - start) / duration);
          const value = Math.round(target * easeOutQuart(t));
          el.textContent = value.toLocaleString();
          if (t < 1) requestAnimationFrame(step);
        };
        requestAnimationFrame(step);
        io.unobserve(el);
      });
    }, { threshold: 0.6 });
    els.forEach((el) => io.observe(el));
  };

  /* GSAP ScrollTrigger — landing choreography.
     Loads only if GSAP global present + motion allowed + on a landing page.
     Fail-open: if GSAP missing, elements render at natural CSS state (visible). */
  const gsapLanding = () => {
    const gsap = window.gsap;
    const ScrollTrigger = window.ScrollTrigger;
    if (!gsap || !ScrollTrigger) return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    if (!document.querySelector('.lp')) return;

    gsap.registerPlugin(ScrollTrigger);
    /* instant-nav safety: clear triggers from a previous page */
    ScrollTrigger.getAll().forEach((t) => t.kill());

    const OUT = 'power2.out';

    /* (a) Hero text — stagger children in on load */
    const heroBits = document.querySelectorAll(
      '.lp-hero-text .lp-eyebrow, .lp-hero-text > h1, .lp-hero-text .lp-sub,' +
      '.lp-hero-text .highlight, .lp-hero-text .lp-cta'
    );
    const introTweens = [];
    if (heroBits.length) {
      introTweens.push(gsap.from(heroBits, {
        y: 0, duration: 0.5, ease: OUT,
        stagger: 0.09, delay: 0.05, clearProps: 'all',
      }));
    }

    /* (b) Hero diagram card — fade up on load */
    const card = document.querySelector('.lp-hero-card');
    if (card) {
      introTweens.push(gsap.from(card, {
        y: 0, duration: 0.6, ease: OUT,
        delay: 0.25, clearProps: 'all',
      }));
    }

    /* Safety net (truly fail-open): an on-load from() holds the above-the-fold hero
       at opacity:0 until the rAF ticker advances it — but rAF is PAUSED in a
       backgrounded tab, so the hero could stay blank for a never-focused tab, a
       crawler, or a prerender/screenshot. setTimeout still fires in the background,
       so force any unfinished intro tween to its visible end-state. No-op once the
       (sub-second) entrance has already completed in a normal foreground load. */
    setTimeout(() => {
      introTweens.forEach((tw) => { if (tw && tw.progress() < 1) tw.progress(1); });
      const introEls = [...heroBits, card].filter(Boolean);
      if (introEls.length) gsap.set(introEls, { clearProps: 'all' });
    }, 1600);

    /* Only animate blocks that start BELOW the fold; anything already in the
       initial viewport (e.g. the short ecosystem landing) stays visible, so a
       flaky/late ScrollTrigger or a re-init can never strand it at opacity:0. */
    const belowFold = (el) => el.getBoundingClientRect().top > (window.innerHeight || 800) * 0.9;

    /* (c) Section eyebrows (FEATURES / TUTORIALS) — slide in from left on scroll */
    gsap.utils.toArray('.lp-section .lp-eyebrow').forEach((eb) => {
      if (!belowFold(eb)) return;
      gsap.from(eb, {
        scrollTrigger: { trigger: eb, start: 'top 88%' },
        x: -16, duration: 0.45, ease: OUT, clearProps: 'all',
      });
    });

    /* (d) Cards — staggered fade-up per grid on scroll */
    gsap.utils.toArray('.lp-grid-4').forEach((grid) => {
      const cards = grid.querySelectorAll('.lp-card');
      if (!cards.length) return;
      if (!belowFold(grid)) return;   // in view at setup → keep visible, don't hide
      gsap.from(cards, {
        scrollTrigger: { trigger: grid, start: 'top 85%' },
        y: 26, duration: 0.5, ease: OUT,
        stagger: 0.08, clearProps: 'all',
      });
    });

    /* (e) Feature columns — staggered fade-up on scroll */
    const featRow = document.querySelector('.lp-feature-row');
    if (featRow && belowFold(featRow)) {
      const feats = featRow.querySelectorAll('.lp-feat');
      if (feats.length) {
        gsap.from(feats, {
          scrollTrigger: { trigger: featRow, start: 'top 88%' },
          y: 22, duration: 0.5, ease: OUT,
          stagger: 0.08, clearProps: 'all',
        });
      }
    }

    /* (f) Footer bar — fade up on scroll */
    const footer = document.querySelector('.lp-footer-bar');
    if (footer && belowFold(footer)) {
      gsap.from(footer, {
        scrollTrigger: { trigger: footer, start: 'top 92%' },
        y: 16, duration: 0.5, ease: OUT, clearProps: 'all',
      });
    }

    ScrollTrigger.refresh();

    /* Fail-open safety net for the scroll-revealed blocks (cards / features /
       footer / eyebrows). gsap.from(...{scrollTrigger}) parks them at opacity:0
       until the trigger fires. On a SHORT landing they already sit in the initial
       viewport, so if ScrollTrigger doesn't fire on load (backgrounded tab,
       prerender, headless screenshot, or odd scroll metrics) they'd stay blank.
       After a beat, force any still-hidden, in-view block to its visible state.
       Below-the-fold blocks on a long page are untouched — they reveal on scroll
       as designed (a working foreground load makes this a no-op). */
    setTimeout(() => {
      const vh = window.innerHeight || 800;
      document.querySelectorAll(
        '.lp-card, .lp-feat, .lp-footer-bar, .lp-section .lp-eyebrow'
      ).forEach((el) => {
        const r = el.getBoundingClientRect();
        const inView = r.top < vh && r.bottom > 0;
        if (inView && parseFloat(getComputedStyle(el).opacity) < 0.5) {
          gsap.set(el, { clearProps: 'opacity,transform' });
          el.style.opacity = '';
          el.style.transform = '';
        }
      });
    }, 1700);
  };

  /* Disentanglement animation (iter 50) — the hero centrepiece.
     A cloud of mixed-colour dots (entangled latent z) sorts itself into
     three clean factor clusters (Cell type / Condition / Batch effect),
     then re-mixes. Loops. Represents M3's factorised embeddings. */
  let disentTl = null;
  const disentanglement = () => {
    const svg = document.querySelector('.lp-disent svg');
    if (!svg) return;
    const dotsG = svg.querySelector('.disent-dots');
    if (!dotsG) return;

    /* instant-nav safety: clear prior dots + timeline */
    if (disentTl) { disentTl.kill(); disentTl = null; }
    dotsG.innerHTML = '';

    const NS = 'http://www.w3.org/2000/svg';
    const lanes = [
      { x: 68,  color: '#7c3aed' },  // Cell type   — violet  (biological)
      { x: 178, color: '#e11d48' },  // Condition 1 — rose
      { x: 288, color: '#fb7185' },  // Condition 2 — salmon (condition family)
      { x: 398, color: '#64748b' },  // Batch effect — slate  (removed in correction)
    ];
    const PER = 5;                   // fewer dots
    const R = 6;                     // bigger dots
    const blobCx = 233, blobCy = 116, blobR = 52;
    const laneTop = 74, laneBot = 206;
    const rand = (a, b) => a + Math.random() * (b - a);

    const dots = [];
    lanes.forEach((lane, li) => {
      for (let i = 0; i < PER; i++) {
        const ang = Math.random() * Math.PI * 2;
        const rr = Math.sqrt(Math.random()) * blobR;
        const ex = blobCx + Math.cos(ang) * rr;
        const ey = blobCy + Math.sin(ang) * rr * 0.82;
        const dx = lane.x + rand(-14, 14);
        const dy = rand(laneTop, laneBot);
        const c = document.createElementNS(NS, 'circle');
        c.setAttribute('cx', ex.toFixed(1));
        c.setAttribute('cy', ey.toFixed(1));
        c.setAttribute('r', String(R));
        c.setAttribute('fill', lane.color);
        c.__ex = ex; c.__ey = ey; c.__dx = dx; c.__dy = dy; c.__lane = li;
        dotsG.appendChild(c);
        dots.push(c);
      }
    });
    const batchDots = dots.filter((d) => d.__lane === 3);

    const laneLabels = svg.querySelectorAll('.disent-label');
    const axes = svg.querySelectorAll('.disent-axis line');
    const zLabel = svg.querySelector('.disent-z');
    const batchNote = svg.querySelector('.disent-batch-note');
    const batchLabel = laneLabels[3];
    const batchAxis = axes[3];
    const gsap = window.gsap;

    /* Fallback: no GSAP or reduced motion → rest in disentangled state. */
    if (!gsap || window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      dots.forEach((c) => { c.setAttribute('cx', c.__dx.toFixed(1)); c.setAttribute('cy', c.__dy.toFixed(1)); });
      laneLabels.forEach((l) => (l.style.opacity = 1));
      axes.forEach((l) => (l.style.opacity = 1));
      if (zLabel) zLabel.style.opacity = 0;
      return;
    }

    const E = 'power2.inOut';
    gsap.set(laneLabels, { opacity: 0 });
    gsap.set(axes, { opacity: 0, transformOrigin: 'center', scaleX: 0.4 });
    gsap.set(zLabel, { opacity: 1 });
    gsap.set(batchNote, { opacity: 0 });

    disentTl = gsap.timeline({ repeat: -1 });

    /* ── beat 1: hold entangled ── */
    disentTl.to({}, { duration: 0.9 });

    /* ── beat 2: DISENTANGLE into 4 factor clusters ── */
    disentTl.to(zLabel, { opacity: 0, duration: 0.3 }, 'sep');
    disentTl.to(dots, {
      duration: 1.5, ease: E, stagger: 0.015,
      attr: { cx: (i, el) => el.__dx, cy: (i, el) => el.__dy },
    }, 'sep');
    disentTl.to(axes, { opacity: 1, scaleX: 1, duration: 0.5, ease: E }, 'sep+=0.8');
    disentTl.to(laneLabels, { opacity: 1, duration: 0.4, stagger: 0.05 }, 'sep+=0.9');

    /* ── beat 3: hold disentangled ── */
    disentTl.to({}, { duration: 0.7 });

    /* ── beat 4: BATCH CORRECTION — batch cluster fades + drifts away ── */
    disentTl.to(batchDots, { opacity: 0.16, y: 18, duration: 0.6, ease: E }, 'rem');
    disentTl.to([batchLabel, batchAxis], { opacity: 0.28, duration: 0.4 }, 'rem');
    disentTl.to(batchNote, { opacity: 1, duration: 0.4 }, 'rem+=0.25');

    /* ── beat 5: hold corrected ── */
    disentTl.to({}, { duration: 1.0 });

    /* ── beat 6: RE-ENTANGLE (restore batch, merge all) ── */
    disentTl.to(batchNote, { opacity: 0, duration: 0.3 }, 'mer');
    disentTl.to(batchDots, { opacity: 1, y: 0, duration: 0.4, ease: E }, 'mer');
    disentTl.to([batchLabel, batchAxis], { opacity: 0, duration: 0.3 }, 'mer');
    disentTl.to(laneLabels, { opacity: 0, duration: 0.3 }, 'mer');
    disentTl.to(axes, { opacity: 0, scaleX: 0.4, duration: 0.3 }, 'mer');
    disentTl.to(dots, {
      duration: 1.4, ease: E, stagger: 0.015,
      attr: { cx: (i, el) => el.__ex, cy: (i, el) => el.__ey },
    }, 'mer');
    disentTl.to(zLabel, { opacity: 1, duration: 0.4 }, 'mer+=0.7');
  };

  /* ── shared SVG helpers for the task-flow ── */
  const TF_NS = 'http://www.w3.org/2000/svg';
  const tfMk = (t, a) => { const e = document.createElementNS(TF_NS, t); for (const k in a) e.setAttribute(k, a[k]); return e; };
  const tfScene = () => { const e = tfMk('g', { class: 'tf-scene', opacity: 0 }); return e; };
  const tfLabel = (x, y, str, anchor, size, fill, weight) => {
    const t = tfMk('text', { x, y, 'text-anchor': anchor || 'start', 'font-size': size || 8.5, fill: fill || 'currentColor' });
    if (weight) t.setAttribute('font-weight', weight);
    t.textContent = str; return t;
  };
  /* organic blob path (smooth closed curve around a centre) */
  const tfBlob = (cx, cy, rx, ry) => {
    const n = 12, pts = [];
    for (let i = 0; i < n; i++) { const a = (i / n) * Math.PI * 2; const k = 0.93 + Math.random() * 0.12; pts.push([cx + Math.cos(a) * rx * k, cy + Math.sin(a) * ry * k]); }
    let d = `M ${pts[0][0].toFixed(1)} ${pts[0][1].toFixed(1)} `;
    for (let i = 0; i < n; i++) {
      const p0 = pts[(i - 1 + n) % n], p1 = pts[i], p2 = pts[(i + 1) % n], p3 = pts[(i + 2) % n];
      const c1x = p1[0] + (p2[0] - p0[0]) / 6, c1y = p1[1] + (p2[1] - p0[1]) / 6;
      const c2x = p2[0] - (p3[0] - p1[0]) / 6, c2y = p2[1] - (p3[1] - p1[1]) / 6;
      d += `C ${c1x.toFixed(1)} ${c1y.toFixed(1)}, ${c2x.toFixed(1)} ${c2y.toFixed(1)}, ${p2[0].toFixed(1)} ${p2[1].toFixed(1)} `;
    }
    return d + 'Z';
  };

  /* right-side legend; appended to scene and registered as a fade-in morph */
  const tfLegend = (morphs, sceneG, x, yTop, rows, immediate) => {
    const g = tfMk('g', {});
    let y = yTop;
    rows.forEach((r) => {
      if (r.title) { g.appendChild(tfLabel(x, y, r.title, 'start', 9, 'currentColor', '700')); y += r.gap || 17; return; }
      if (r.swatch === 'tri') g.appendChild(tfMk('polygon', { points: `${x + 5},${y - 8} ${x + 10},${y} ${x},${y}`, fill: r.color || '#64748b' }));
      else if (r.swatch === 'ring') { const rc = tfMk('circle', { cx: x + 5, cy: y - 3, r: 4.5, fill: 'none', stroke: r.color || 'currentColor', 'stroke-width': 1.4 }); if (r.dashed) rc.setAttribute('stroke-dasharray', '2.5 1.8'); g.appendChild(rc); }
      else if (r.swatch === 'striped') { g.appendChild(tfMk('circle', { cx: x + 5, cy: y - 3, r: 5, fill: r.color })); g.appendChild(tfMk('line', { x1: x + 1.5, y1: y, x2: x + 8.5, y2: y - 7, stroke: '#fff', 'stroke-width': 1.1 })); }
      else if (r.swatch === 'sq') g.appendChild(tfMk('rect', { x, y: y - 8, width: 10, height: 10, rx: 1, fill: r.color }));
      else if (r.swatch === 'dot') g.appendChild(tfMk('circle', { cx: x + 5, cy: y - 3, r: 5, fill: r.color }));
      g.appendChild(tfLabel(x + 18, y, r.text, 'start', 8.5));
      y += r.gap || 15;
    });
    sceneG.appendChild(g);
    // immediate → legend rides the scene fade-in (visible from the start), no delayed reveal
    if (!immediate) morphs.push({ el: g, from: { opacity: 0 }, to: { opacity: 1 }, dur: 0.4, delay: 0.5 });
    return g;
  };

  /* Task-flow showcase (iter 54) — six downstream tasks, smooth in-place
     morphs + per-scene legends on the right. No arrows. GSAP master timeline. */
  const taskFlow = () => {
    const wrap = document.querySelector('.taskflow');
    // instant-nav: a previous page's hero card may have left its perpetual
    // repeat:-1 timeline running on now-detached nodes (burning rAF forever).
    // Kill it when the current page has no task-flow card.
    if (!wrap) { if (window.__tfTl) { window.__tfTl.kill(); window.__tfTl = null; } return; }
    const scenesG = wrap.querySelector('.tf-scenes');
    const headNum = wrap.querySelector('.tf-num');
    const headTitle = wrap.querySelector('.tf-title');
    const dots = wrap.querySelectorAll('.tf-dots span');
    if (!scenesG) return;
    if (window.__tfTl) { window.__tfTl.kill(); window.__tfTl = null; }
    scenesG.innerHTML = '';

    const mk = tfMk, label = tfLabel;
    const BLUES = ['#dbeafe', '#93c5fd', '#60a5fa', '#3b82f6', '#1d4ed8'];
    const CT = ['#3b82f6', '#10b981', '#ef4444'];
    const COND_A = '#f59e0b', COND_B = '#3b82f6', UNK = '#cbd5e1';
    const rnd = (a) => a[Math.floor(Math.random() * a.length)];
    const PERSON = 'M11 2.4a3.3 3.3 0 1 1 0 6.6 3.3 3.3 0 0 1 0-6.6zm0 8.1c3.7 0 6.6 1.8 6.6 4.1V17H4.4v-2.4c0-2.3 2.9-4.1 6.6-4.1z';
    const person = (cx, cy, h, fill) => {
      const s = h / 19;
      const outer = mk('g', {});
      const inner = mk('g', { transform: `translate(${(cx - 11 * s).toFixed(1)},${(cy - 9.5 * s).toFixed(1)}) scale(${s.toFixed(3)})` });
      const p = mk('path', { d: PERSON, fill });
      inner.appendChild(p); outer.appendChild(inner); outer.__path = p;
      return outer;
    };
    const cell = (cx, cy, color, tri, striped) => {
      const g = mk('g', {});
      if (tri) g.appendChild(mk('polygon', { points: `${cx},${cy - 7} ${cx + 6.4},${cy + 5.5} ${cx - 6.4},${cy + 5.5}`, fill: color }));
      else g.appendChild(mk('circle', { cx, cy, r: 6, fill: color }));
      if (striped) {
        g.appendChild(mk('line', { x1: cx - 3.5, y1: cy + 3, x2: cx + 3, y2: cy - 3.5, stroke: '#fff', 'stroke-width': 1.1, opacity: 0.85 }));
        g.appendChild(mk('line', { x1: cx - 0.5, y1: cy + 4.5, x2: cx + 4.5, y2: cy - 0.5, stroke: '#fff', 'stroke-width': 1.1, opacity: 0.85 }));
      }
      return g;
    };

    const reg = (arr, el, from, to, dur, delay) => arr.push({ el, from, to, dur: dur || 1.3, delay: delay || 0 });

    /* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       Per-project scene sets. Each builder returns a fresh `scenes[]` array so the
       same task-flow card can switch between M3 / scMultiBench / Matilda demos.
       The active set is chosen by `.taskflow[data-taskflow]` and, on the ecosystem
       landing, by the `.tf-switch` segmented control.
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
    const BUILDERS = {};

    BUILDERS.m3 = () => {
    const scenes = [];

    /* ① Factorised DR — big feature matrix SHRINKS + factorises into low-dim strips */
    (() => {
      const g = tfScene(); const m = [];
      const stripCol = ['#7c3aed', '#e11d48', '#fb7185', '#64748b'];
      const stripY = [58, 84, 110, 136];     // tighter row spacing (26px)
      const PERROW = 6, KEEP = 4 * PERROW;   // fewer per row
      for (let i = 0; i < 54; i++) {
        const col = i % 9, row = Math.floor(i / 9);
        const fx = 44 + col * 15, fy = 46 + row * 15;
        const r = mk('rect', { x: fx, y: fy, width: 13, height: 13, rx: 1, fill: rnd(BLUES) });
        g.appendChild(r);
        if (i < KEEP) {
          const strip = Math.floor(i / PERROW), pos = i % PERROW;
          reg(m, r, { attr: { x: fx, y: fy, width: 13, height: 13, fill: rnd(BLUES) } }, { attr: { x: 80 + pos * 20, y: stripY[strip], width: 18, height: 18, fill: stripCol[strip] } }, 1.3, i * 0.012);
        } else {
          reg(m, r, { attr: { width: 13, height: 13 }, opacity: 1 }, { attr: { width: 3, height: 3 }, opacity: 0 }, 1.0, (i - KEEP) * 0.01);
        }
      }
      tfLegend(m, g, 300, 56, [
        { title: 'Factors' },
        { swatch: 'sq', color: '#7c3aed', text: 'Cell type' },
        { swatch: 'sq', color: '#e11d48', text: 'Condition 1' },
        { swatch: 'sq', color: '#fb7185', text: 'Condition 2' },
        { swatch: 'sq', color: '#64748b', text: 'Batch' },
      ]);
      scenes.push({ num: 1, title: 'Factorised dimension reduction', g, morphs: m });
    })();

    /* ② Batch correction — cells cluster by (cell type × condition); organic blobs */
    (() => {
      const g = tfScene(); const m = [];
      // 6 groups = 3 cell types × 2 conditions, laid out like the paper
      const groups = [
        { color: CT[2], striped: false, cx: 70,  cy: 58 },   // red  · cond 1
        { color: CT[2], striped: true,  cx: 162, cy: 62 },   // red  · cond 2
        { color: CT[1], striped: true,  cx: 248, cy: 84 },   // green· cond 2
        { color: CT[1], striped: false, cx: 250, cy: 158 },  // green· cond 1
        { color: CT[0], striped: true,  cx: 96,  cy: 144 },  // blue · cond 2
        { color: CT[0], striped: false, cx: 178, cy: 158 },  // blue · cond 1
      ];
      groups.forEach((grp) => {
        const blob = mk('path', { d: tfBlob(grp.cx, grp.cy, 34, 30), fill: grp.color, opacity: 0 });
        g.appendChild(blob);
        reg(m, blob, { opacity: 0 }, { opacity: 0.15 }, 0.6, 1.2);
        const N = 5;
        for (let i = 0; i < N; i++) {
          const ang = (i / N) * Math.PI * 2 + Math.random() * 0.6;
          const rr = 6 + Math.random() * 15;
          const tx = grp.cx + Math.cos(ang) * rr, ty = grp.cy + Math.sin(ang) * rr * 0.9;
          const sx = 40 + Math.random() * 220, sy = 36 + Math.random() * 150;
          const cg = cell(sx, sy, grp.color, i % 2 === 0, grp.striped);
          g.appendChild(cg);
          reg(m, cg, { x: 0, y: 0 }, { x: tx - sx, y: ty - sy }, 1.5, Math.random() * 0.35);
        }
      });
      tfLegend(m, g, 302, 40, [
        { title: 'Cell type' },
        { swatch: 'dot', color: CT[0], text: 'Type A' }, { swatch: 'dot', color: CT[1], text: 'Type B' }, { swatch: 'dot', color: CT[2], text: 'Type C' },
        { title: 'Batch' }, { swatch: 'ring', text: 'Batch 1' }, { swatch: 'tri', text: 'Batch 2' },
        { title: 'Condition' }, { swatch: 'dot', color: '#94a3b8', text: 'Cond 1' }, { swatch: 'striped', color: '#94a3b8', text: 'Cond 2' },
      ]);
      scenes.push({ num: 2, title: 'Condition-aware batch correction', g, morphs: m });
    })();

    /* ③ Mosaic imputation — an ENTIRE missing modality is generated */
    (() => {
      const g = tfScene(); const m = [];
      // no legend on this scene → centre the grid in the full 460-wide stage
      const cols = 8, c0 = 15, gap = 2, gridW = cols * c0 + (cols - 1) * gap;
      const x0 = Math.round(230 - gridW / 2), y1 = 44, y2 = 116;
      for (let r = 0; r < 3; r++) for (let c = 0; c < cols; c++)
        g.appendChild(mk('rect', { x: x0 + c * (c0 + gap), y: y1 + r * (c0 + gap), width: c0, height: c0, rx: 1, fill: rnd(BLUES) }));
      for (let r = 0; r < 3; r++) for (let c = 0; c < cols; c++) {
        const rect = mk('rect', { x: x0 + c * (c0 + gap), y: y2 + r * (c0 + gap), width: c0, height: c0, rx: 1, fill: rnd(BLUES) });
        g.appendChild(rect);
        reg(m, rect, { attr: { 'fill-opacity': 0, stroke: '#94a3b8', 'stroke-width': 0.8, 'stroke-dasharray': '2 1.5' } }, { attr: { 'fill-opacity': 1, 'stroke-width': 0 } }, 0.5, c * 0.09 + r * 0.04);
      }
      g.appendChild(label(x0, y1 - 6, 'Modality 1 (observed)', 'start', 9));
      g.appendChild(label(x0, y2 - 6, 'Modality 2 (missing → imputed)', 'start', 9));
      scenes.push({ num: 3, title: 'Mosaic integration & imputation', g, morphs: m });
    })();

    /* ④ Patient inference — unknown (grey) donors get inferred to a condition */
    (() => {
      const g = tfScene(); const m = [];
      const plan = [COND_A, COND_A, COND_A, COND_B, COND_B, COND_B, UNK, UNK, UNK, UNK];
      const infer = [null, null, null, null, null, null, COND_B, COND_A, COND_B, COND_A];
      const cols = 5, gx = 56, x0 = 230 - (cols - 1) * gx / 2 - 70;
      plan.forEach((col, i) => {
        const cx = x0 + (i % cols) * gx, cy = 70 + Math.floor(i / cols) * 64;
        const pe = person(cx, cy, 44, col);
        g.appendChild(pe);
        if (infer[i]) reg(m, pe.__path, { attr: { fill: UNK } }, { attr: { fill: infer[i] } }, 0.7, 0.7 + (i - 6) * 0.2);
      });
      tfLegend(m, g, 300, 72, [
        { title: 'Condition' },
        { swatch: 'dot', color: COND_A, text: 'Diseased' },
        { swatch: 'dot', color: COND_B, text: 'Healthy' },
        { swatch: 'dot', color: UNK, text: 'Unknown' },
      ]);
      scenes.push({ num: 4, title: 'Patient-level condition inference', g, morphs: m });
    })();

    /* ⑤ Sample generation — input people stay, generated people grow in */
    (() => {
      const g = tfScene(); const m = [];
      const genCol = [COND_B, COND_A];
      const cols = 6, rows = 3, gx = 62, x0 = 230 - (cols - 1) * gx / 2;
      let idx = 0;
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
        const cx = x0 + c * gx, cy = 54 + r * 50;
        const pe = person(cx, cy, 40, genCol[(r + c) % 2]);
        g.appendChild(pe);
        if (idx < 4) reg(m, pe, { opacity: 1, scale: 1, svgOrigin: `${cx} ${cy}` }, { opacity: 1, scale: 1, svgOrigin: `${cx} ${cy}` }, 0.3, 0);
        else reg(m, pe, { opacity: 0, scale: 0, svgOrigin: `${cx} ${cy}` }, { opacity: 1, scale: 1, svgOrigin: `${cx} ${cy}` }, 0.5, (idx - 4) * 0.05);
        idx++;
      }
      scenes.push({ num: 5, title: 'Patient-level sample generation', g, morphs: m });
    })();

    /* ⑥ Attribution — highlight cells / genes / cell types important for disease */
    (() => {
      const g = tfScene(); const m = [];
      const REFCT = ['#6366f1', '#14b8a6', '#fb7185'];   // refined indigo / teal / coral
      const IMP = '#be123c';                              // elegant crimson rings
      const centers = [[78, 70], [162, 58], [118, 138]];
      // deterministic two-row layout: every pair ≥17px apart so cells never
      // touch and an emphasis ring never clips a neighbour
      const off = [[-17, -11], [0, -15], [17, -11], [-17, 9], [0, 13], [17, 9]];
      const dots = [];
      for (let c = 0; c < 3; c++) for (let i = 0; i < 6; i++) {
        const cx = centers[c][0] + off[i][0], cy = centers[c][1] + off[i][1];
        const d = mk('circle', { cx, cy, r: 5.5, fill: REFCT[c] });
        g.appendChild(d); dots.push(d);
      }
      g.appendChild(label(118, 184, 'Cells', 'middle', 9));
      // emphasis = a crimson ring drawn AROUND select cells; cell size stays equal
      [2, 6, 11, 16].forEach((k, j) => {
        const d = dots[k]; if (!d) return;
        const cx = +d.getAttribute('cx'), cy = +d.getAttribute('cy');
        const halo = mk('circle', { cx, cy, r: 10, fill: 'none', stroke: IMP, 'stroke-width': 2.2, opacity: 0 });
        g.appendChild(halo);
        reg(m, halo, { opacity: 0, scale: 1.4, svgOrigin: `${cx} ${cy}` }, { opacity: 1, scale: 1, svgOrigin: `${cx} ${cy}` }, 0.5, 0.7 + j * 0.1);
      });
      // dashed cell-type region around the coral cluster
      const region = mk('ellipse', { cx: centers[2][0], cy: centers[2][1], rx: 38, ry: 32, fill: 'none', stroke: IMP, 'stroke-width': 1.6, 'stroke-dasharray': '5 3', opacity: 0 });
      g.appendChild(region); reg(m, region, { opacity: 0 }, { opacity: 1 }, 0.5, 1.0);
      for (let i = 0; i < 9; i++) {
        const rr = mk('rect', { x: 40 + i * 13, y: 196, width: 11, height: 11, rx: 1.5, fill: '#e2e8f0' });
        g.appendChild(rr);
        if ([1, 4, 7].includes(i)) reg(m, rr, { attr: { fill: '#e2e8f0' } }, { attr: { fill: IMP } }, 0.4, 0.8 + i * 0.04);
      }
      g.appendChild(label(40, 192, 'Genes', 'start', 9));
      // disease patient keeps the amber 'Diseased' colour from the earlier scenes
      g.appendChild(person(262, 60, 54, COND_A));
      g.appendChild(label(262, 102, 'Disease', 'middle', 11, 'currentColor', '600'));
      tfLegend(m, g, 300, 108, [
        { title: 'Important for' },
        { title: 'prediction:', gap: 18 },
        { swatch: 'dot', color: IMP, text: 'Cells' },
        { swatch: 'sq', color: IMP, text: 'Genes' },
        { swatch: 'ring', color: IMP, text: 'Cell Type', dashed: true },
      ]);
      scenes.push({ num: 6, title: 'Multi-resolution attribution', g, morphs: m });
    })();

    return scenes;
    };  /* end BUILDERS.m3 */

    /* ══════════════════════════ Matilda ══════════════════════════ */
    BUILDERS.matilda = () => {
    const scenes = [];

    /* ① Multimodal integration — three modality inputs → arrow → a centred shared latent z.
       Balanced left→centre→right layout (inputs left, z centre, legend right) so the
       frame never goes lopsided the way a lone z + right legend did. */
    (() => {
      const g = tfScene(); const m = [];
      const mods = [
        { color: '#7c3aed', y: 60 },   // RNA
        { color: '#0ea5e9', y: 100 },  // ADT
        { color: '#f59e0b', y: 140 },  // ATAC
      ];
      // three modality input strips on the left (colours are keyed in the legend; no inline text)
      const sN = 4, ssq = 14, sgap = 4, sx0 = 40;
      mods.forEach((mod, mi) => {
        for (let i = 0; i < sN; i++) {
          const rr = mk('rect', { x: sx0 + i * (ssq + sgap), y: mod.y, width: ssq, height: ssq, rx: 2, fill: mod.color, opacity: 0 });
          g.appendChild(rr);
          reg(m, rr, { opacity: 0 }, { opacity: 0.9 }, 0.4, 0.15 + mi * 0.08 + i * 0.05);
        }
      });
      // arrow — fades in WITH the modality strips (not before, in a blank frame)
      const arr1 = mk('line', { x1: 128, y1: 104, x2: 192, y2: 104, stroke: '#94a3b8', 'stroke-width': 2.2, opacity: 0 });
      g.appendChild(arr1);
      const arh1 = mk('polygon', { points: '192,99 203,104 192,109', fill: '#94a3b8', opacity: 0 });
      g.appendChild(arh1);
      reg(m, arr1, { opacity: 0 }, { opacity: 1 }, 0.4, 0.25);
      reg(m, arh1, { opacity: 0 }, { opacity: 1 }, 0.4, 0.3);
      // shared latent z — centred in the frame (column centre ≈ x230)
      const latX = 224;
      const lat = mk('g', {});
      for (let k = 0; k < 6; k++) lat.appendChild(mk('rect', { x: latX, y: 64 + k * 16, width: 14, height: 14, rx: 2, fill: '#6d28d9' }));
      g.appendChild(lat);
      reg(m, lat, { opacity: 0, scale: 0.5, svgOrigin: (latX + 7) + ' 112' }, { opacity: 1, scale: 1, svgOrigin: (latX + 7) + ' 112' }, 0.7, 0.55);
      const zlab = label(latX + 7, 56, 'z', 'middle', 11, '#6d28d9', '700'); zlab.setAttribute('opacity', 0);
      g.appendChild(zlab);
      reg(m, zlab, { opacity: 0 }, { opacity: 1 }, 0.5, 0.85);
      // legend fades in with the rest of the scene (synced, not ahead of the right side)
      tfLegend(m, g, 300, 58, [
        { title: 'Modalities' },
        { swatch: 'sq', color: '#7c3aed', text: 'RNA' },
        { swatch: 'sq', color: '#0ea5e9', text: 'ADT' },
        { swatch: 'sq', color: '#f59e0b', text: 'ATAC' },
        { title: 'VAE encoder' },
        { swatch: 'sq', color: '#6d28d9', text: 'Shared latent z' },
      ], false);
      scenes.push({ num: 1, title: 'Multimodal integration', g, morphs: m });
    })();

    /* ② Data simulation — a few real cells → an arrow → many neatly-tiled simulated cells */
    (() => {
      const g = tfScene(); const m = [];
      // a few real "seed" cells on the left, one per cell type — these fade in first
      CT.forEach((col, i) => {
        const rc = mk('circle', { cx: 60, cy: 78 + i * 36, r: 7, fill: col, opacity: 0 });
        g.appendChild(rc);
        reg(m, rc, { opacity: 0 }, { opacity: 1 }, 0.4, 0.1 + i * 0.05);
      });
      // arrow — appears WITH the real cells (not before), then the grid grows from it
      const arr2 = mk('line', { x1: 92, y1: 114, x2: 150, y2: 114, stroke: '#94a3b8', 'stroke-width': 2.4, opacity: 0 });
      g.appendChild(arr2);
      const arh2 = mk('polygon', { points: '150,109 161,114 150,119', fill: '#94a3b8', opacity: 0 });
      g.appendChild(arh2);
      reg(m, arr2, { opacity: 0 }, { opacity: 1 }, 0.4, 0.2);
      reg(m, arh2, { opacity: 0 }, { opacity: 1 }, 0.4, 0.25);
      // many VAE-simulated cells in a TIDY GRID (1 → many); tile in row by row
      const cols = 5, rows = 5, dx = 22, dy = 22, gx0 = 178, gy0 = 64;
      let idx = 0;
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
        const cx = gx0 + c * dx, cy = gy0 + r * dy;
        const ci = mk('circle', { cx, cy, r: 6, fill: CT[(r + c) % 3] });
        g.appendChild(ci);
        reg(m, ci, { opacity: 0, scale: 0, svgOrigin: cx + ' ' + cy }, { opacity: 1, scale: 1, svgOrigin: cx + ' ' + cy }, 0.45, 0.3 + idx * 0.03);
        idx++;
      }
      // legend matches the actual cells (coloured by type); real vs simulated is shown by the arrow
      tfLegend(m, g, 300, 70, [
        { title: 'Cell type' },
        { swatch: 'dot', color: CT[0], text: 'Type A' },
        { swatch: 'dot', color: CT[1], text: 'Type B' },
        { swatch: 'dot', color: CT[2], text: 'Type C' },
      ]);
      scenes.push({ num: 2, title: 'Data simulation', g, morphs: m });
    })();

    /* ③ Cell-type classification — query cells are coloured by their predicted type */
    (() => {
      const g = tfScene(); const m = [];
      const cols = 5, rows = 3, gx = 42, gy = 44, x0 = 60, y0 = 58;
      let idx = 0;
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
        const cx = x0 + c * gx, cy = y0 + r * gy;
        const col = CT[(c + r) % 3];
        const ci = mk('circle', { cx, cy, r: 6.5, fill: '#cbd5e1' });
        g.appendChild(ci);
        reg(m, ci, { attr: { fill: '#cbd5e1' } }, { attr: { fill: col } }, 0.5, 0.5 + idx * 0.045);
        idx++;
      }
      tfLegend(m, g, 300, 64, [
        { title: 'Predicted' },
        { swatch: 'dot', color: CT[0], text: 'Type A' },
        { swatch: 'dot', color: CT[1], text: 'Type B' },
        { swatch: 'dot', color: CT[2], text: 'Type C' },
      ]);
      scenes.push({ num: 3, title: 'Cell-type classification', g, morphs: m });
    })();

    /* ④ Feature selection — integrated gradients highlight markers across modalities */
    (() => {
      const g = tfScene(); const m = [];
      const IMP = '#be123c';
      const rows = [
        { name: 'RNA',  y: 56,  imp: [2, 5, 8] },
        { name: 'ADT',  y: 104, imp: [1, 4] },
        { name: 'ATAC', y: 152, imp: [0, 3, 6, 9] },
      ];
      const N = 11, sq = 13, gap = 4, x0 = 70;
      rows.forEach((row) => {
        g.appendChild(label(x0 - 30, row.y + 10, row.name, 'start', 8.5, 'currentColor', '700'));
        for (let i = 0; i < N; i++) {
          const fx = x0 + i * (sq + gap);
          const r = mk('rect', { x: fx, y: row.y, width: sq, height: sq, rx: 1.5, fill: '#e2e8f0' });
          g.appendChild(r);
          if (row.imp.includes(i)) reg(m, r, { attr: { fill: '#e2e8f0' } }, { attr: { fill: IMP } }, 0.4, 0.6 + i * 0.05);
        }
      });
      tfLegend(m, g, 300, 84, [
        { title: 'Integrated' },
        { title: 'gradients', gap: 18 },
        { swatch: 'sq', color: IMP, text: 'Important' },
        { swatch: 'sq', color: '#e2e8f0', text: 'Other' },
      ]);
      scenes.push({ num: 4, title: 'Feature selection', g, morphs: m });
    })();

    return scenes;
    };  /* end BUILDERS.matilda */

    /* ══════════════════════════ scMultiBench ══════════════════════════
       run → evaluate → plot, the scIB-style benchmark pipeline. */
    BUILDERS.multibench = () => {
    const scenes = [];
    const VIO = '#7c3aed', VIO_SOFT = '#a78bfa', COLB = '#0ea5e9';

    /* ① Run — pick one of 40+ integration methods, run it → an integrated embedding */
    (() => {
      const g = tfScene(); const m = [];
      g.appendChild(label(34, 40, '40+ methods', 'start', 9, 'currentColor', '700'));
      // a real, named list of integration methods — one is picked (highlighted) and run
      const names = ['totalVI', 'Seurat', 'Harmony', 'scVI', 'LIGER', '+ 35 more'];
      const SEL = 0;
      const rw = 108, rh = 19, rgap = 5, rx = 34, ry0 = 50;
      names.forEach((nm, i) => {
        const y = ry0 + i * (rh + rgap), sel = (i === SEL);
        const chip = mk('rect', { x: rx, y, width: rw, height: rh, rx: 4, fill: sel ? VIO : '#eef2f7', opacity: 0 });
        g.appendChild(chip);
        const txt = label(rx + 10, y + 13, nm, 'start', 9, sel ? '#fff' : '#475569', sel ? '700' : '500');
        txt.setAttribute('opacity', 0);
        g.appendChild(txt);
        reg(m, chip, { opacity: 0 }, { opacity: 1 }, 0.5, 0.1 + i * 0.06);
        reg(m, txt, { opacity: 0 }, { opacity: 1 }, 0.4, 0.18 + i * 0.06);
      });
      // arrow: run the picked method →
      const arrow = mk('line', { x1: 158, y1: 120, x2: 206, y2: 120, stroke: VIO, 'stroke-width': 2.4, opacity: 0 });
      g.appendChild(arrow);
      const head = mk('polygon', { points: '206,115 216,120 206,125', fill: VIO, opacity: 0 });
      g.appendChild(head);
      reg(m, arrow, { opacity: 0 }, { opacity: 1 }, 0.4, 0.7);
      reg(m, head, { opacity: 0 }, { opacity: 1 }, 0.4, 0.8);
      // → an integrated embedding (cells cluster by type)
      const clusters = [ { col: CT[0], cx: 300, cy: 80 }, { col: CT[1], cx: 352, cy: 150 }, { col: CT[2], cx: 374, cy: 88 } ];
      clusters.forEach((cl, ci) => {
        for (let i = 0; i < 5; i++) {
          const ang = (i / 5) * Math.PI * 2;
          const cx = cl.cx + Math.cos(ang) * 13, cy = cl.cy + Math.sin(ang) * 11;
          const d = mk('circle', { cx, cy, r: 5, fill: cl.col, opacity: 0 });
          g.appendChild(d);
          reg(m, d, { opacity: 0, scale: 0, svgOrigin: cx + ' ' + cy }, { opacity: 1, scale: 1, svgOrigin: cx + ' ' + cy }, 0.5, 1.0 + ci * 0.1 + i * 0.03);
        }
      });
      scenes.push({ num: 1, title: 'Run a method', g, morphs: m });
    })();

    /* ② Evaluate — scIB metric bars grow from the integrated embedding */
    (() => {
      const g = tfScene(); const m = [];
      const COLBIO = '#7c3aed';
      const metrics = [
        { name: 'iLISI', v: 0.78, kind: 'batch' },
        { name: 'kBET',  v: 0.61, kind: 'batch' },
        { name: 'ASW',   v: 0.85, kind: 'bio' },
        { name: 'ARI',   v: 0.72, kind: 'bio' },
        { name: 'NMI',   v: 0.89, kind: 'bio' },
      ];
      const x0 = 100, y0 = 44, bh = 15, gap = 11, maxW = 214;
      metrics.forEach((mt, i) => {
        const y = y0 + i * (bh + gap);
        g.appendChild(label(x0 - 8, y + bh - 4, mt.name, 'end', 9, 'currentColor', '600'));
        g.appendChild(mk('rect', { x: x0, y, width: maxW, height: bh, rx: 3, fill: '#eef2f7' }));
        const bar = mk('rect', { x: x0, y, width: 0, height: bh, rx: 3, fill: mt.kind === 'batch' ? COLB : COLBIO });
        g.appendChild(bar);
        reg(m, bar, { attr: { width: 0 } }, { attr: { width: Math.round(maxW * mt.v) } }, 0.9, 0.2 + i * 0.12);
      });
      // a faded extra row + ellipsis — these are only a few of the scIB metrics scMultiBench computes
      const eY = y0 + metrics.length * (bh + gap);
      g.appendChild(mk('rect', { x: x0, y: eY, width: Math.round(maxW * 0.45), height: bh, rx: 3, fill: '#eef2f7', opacity: 0.6 }));
      g.appendChild(label(x0 - 8, eY + bh - 4, '…', 'end', 12, 'currentColor', '700'));
      tfLegend(m, g, 340, 72, [
        { title: 'scIB metrics' },
        { swatch: 'sq', color: COLB, text: 'Batch removal' },
        { swatch: 'sq', color: COLBIO, text: 'Bio conservation' },
      ]);
      scenes.push({ num: 2, title: 'Evaluate (scIB)', g, morphs: m });
    })();

    /* ③ Plot — methods (rows) × scIB metrics (columns); bubble size = score, top method highlighted */
    (() => {
      const g = tfScene(); const m = [];
      // generic method labels — scMultiBench is a neutral benchmark, not promoting any one tool
      const methods = ['Method 1', 'Method 2', 'Method 3', 'Method 4', 'Method 5'];
      const metrics = ['ARI', 'NMI', 'ASW', 'iLISI', 'kBET', 'cLISI'];
      const nM = methods.length, nMet = metrics.length, x0 = 112, y0 = 60, cw = 36, ch = 25;
      // 'Metrics' axis title (rows are self-labelled Method 1..5)
      g.appendChild(label(x0 + (nMet - 1) * cw / 2, 30, 'Metrics', 'middle', 9, 'currentColor', '700'));
      // column headers = scIB metric names
      metrics.forEach((mt, j) => g.appendChild(label(x0 + j * cw, y0 - 14, mt, 'middle', 7.5, 'currentColor', '600')));
      for (let i = 0; i < nM; i++) {
        // row headers = method names (top method in bold)
        g.appendChild(label(x0 - 14, y0 + i * ch + 3, methods[i], 'end', 8, 'currentColor', i === 0 ? '700' : '500'));
        for (let j = 0; j < nMet; j++) {
          const cx = x0 + j * cw, cy = y0 + i * ch;
          const score = 0.35 + ((i * 7 + j * 3) % 10) / 15;
          const rr = 4 + score * 7;
          const best = (i === 0);
          const c = mk('circle', { cx, cy, r: rr, fill: best ? '#6d28d9' : VIO_SOFT, opacity: best ? 1 : 0.8 });
          g.appendChild(c);
          reg(m, c, { opacity: 0, scale: 0, svgOrigin: cx + ' ' + cy }, { opacity: best ? 1 : 0.8, scale: 1, svgOrigin: cx + ' ' + cy }, 0.5, 0.15 + (i * nMet + j) * 0.022);
        }
      }
      // ellipsis column — many more scIB metrics than the six shown
      const eX = x0 + nMet * cw;
      g.appendChild(label(eX, y0 - 14, '…', 'middle', 12, 'currentColor', '700'));
      for (let i = 0; i < nM; i++) g.appendChild(mk('circle', { cx: eX, cy: y0 + i * ch, r: 4, fill: VIO_SOFT, opacity: 0.3 }));
      tfLegend(m, g, 352, 64, [
        { title: 'Bubble = score' },
        { swatch: 'dot', color: '#6d28d9', text: 'Top method' },
        { swatch: 'dot', color: VIO_SOFT, text: 'Others' },
      ]);
      scenes.push({ num: 3, title: 'Rank & plot results', g, morphs: m });
    })();

    return scenes;
    };  /* end BUILDERS.multibench */

    /* ── Render the active scene-set; dots are rebuilt to match its length ── */
    const render = (key) => {
      if (!BUILDERS[key]) key = 'm3';
      if (window.__tfTl) { window.__tfTl.kill(); window.__tfTl = null; }
      scenesG.innerHTML = '';
      const scenes = BUILDERS[key]();
      scenes.forEach((s) => scenesG.appendChild(s.g));

      const dotsWrap = wrap.querySelector('.tf-dots');
      if (dotsWrap) {
        dotsWrap.innerHTML = '';
        scenes.forEach((_, i) => {
          const sp = document.createElement('span');
          if (i === 0) sp.className = 'is-active';
          dotsWrap.appendChild(sp);
        });
      }
      const dots = wrap.querySelectorAll('.tf-dots span');

    const setHead = (i) => {
      headNum.textContent = scenes[i].num;
      headTitle.textContent = scenes[i].title;
      dots.forEach((d, j) => d.classList.toggle('is-active', j === i));
    };

    const gsap = window.gsap;
    if (!gsap) { scenes[0].g.setAttribute('opacity', 1); setHead(0); return; }
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      scenes[0].g.setAttribute('opacity', 1);
      scenes[0].morphs.forEach((mo) => gsap.set(mo.el, mo.to));
      setHead(0); return;
    }

    setHead(0);
    const master = gsap.timeline({ repeat: -1 });
    window.__tfTl = master;
    const startTimes = [];                   // where each scene begins on the master
    scenes.forEach((s, idx) => {
      startTimes[idx] = master.duration();   // current end = this scene's start time
      const st = gsap.timeline();
      st.call(() => setHead(idx), null, 0);
      st.set(s.g, { opacity: 0 }, 0);
      s.morphs.forEach((mo) => st.set(mo.el, mo.from, 0));
      st.to(s.g, { opacity: 1, duration: 0.4, ease: 'power2.out' }, 0);
      let maxEnd = 0.45;
      s.morphs.forEach((mo) => {
        st.to(mo.el, Object.assign({ duration: mo.dur, ease: 'power2.inOut', delay: mo.delay }, mo.to), 0.45);
        maxEnd = Math.max(maxEnd, 0.45 + mo.delay + mo.dur);
      });
      st.to(s.g, { opacity: 0, duration: 0.45, ease: 'power2.in' }, maxEnd + 3.3); /* +2s hold */
      master.add(st);
    });

    /* Clickable dots: jump to any scene and keep auto-playing from there.
       The master is one declarative timeline, so seeking to a scene's start
       renders that scene's exact state; play() resumes the loop. Dots are
       rebuilt on every render(), so handlers are wired fresh each switch
       (old dot elements — and their listeners — are discarded). */
    dots.forEach((dot, i) => {
      dot.addEventListener('click', () => { setHead(i); master.seek(startTimes[i]).play(); });
    });
    };  /* end render() */

    render((wrap.dataset.taskflow) || 'm3');

    /* Perf: pause the perpetual loop when the card is scrolled off-screen, so it
       isn't burning CPU/rAF the whole time you're reading further down the page. */
    if (!wrap.dataset.visWired && 'IntersectionObserver' in window) {
      wrap.dataset.visWired = '1';
      new IntersectionObserver((ents) => {
        ents.forEach((e) => { const tl = window.__tfTl; if (tl) { e.isIntersecting ? tl.play() : tl.pause(); } });
      }, { threshold: 0.01 }).observe(wrap);
    }

    /* Ecosystem landing only: a segmented control swaps the active scene-set. */
    const switchWrap = document.querySelector('.tf-switch');
    if (switchWrap && !switchWrap.dataset.wired) {
      switchWrap.dataset.wired = '1';
      const btns = switchWrap.querySelectorAll('[data-tf]');
      btns.forEach((btn) => {
        btn.addEventListener('click', () => {
          btns.forEach((b) => {
            const on = b === btn;
            b.classList.toggle('is-active', on);
            b.setAttribute('aria-selected', on ? 'true' : 'false');
          });
          render(btn.dataset.tf);
        });
      });
    }
  };

  /* Sliding TOC indicator — one vertical bar that animates to the active heading,
     instead of a border jumping between items (modern docs style).
     Tracks Material's `.md-nav__link--active`, which the scrollspy toggles. */
  const tocSlider = () => {
    const getCtx = () => {
      const list = document.querySelector('.md-sidebar--secondary .md-nav--secondary > .md-nav__list');
      if (!list) return null;
      let bar = list.querySelector(':scope > .toc-indicator');
      if (!bar) { bar = document.createElement('div'); bar.className = 'toc-indicator'; list.appendChild(bar); }
      return { list, bar };
    };
    let raf = null;
    const move = () => {
      raf = null;
      const ctx = getCtx();
      if (!ctx) return;
      const active = document.querySelector('.md-sidebar--secondary .md-nav__link--active');
      // Keep the bar where it is during the brief moments the scrollspy has no
      // active link (it toggles classes in two steps) — avoids flicker.
      if (!active) return;
      const a = active.getBoundingClientRect(), l = ctx.list.getBoundingClientRect();
      ctx.bar.style.height = Math.round(a.height) + 'px';
      ctx.bar.style.transform = 'translateY(' + Math.round(a.top - l.top) + 'px)';
      ctx.bar.style.opacity = '1';
    };
    const schedule = () => { if (raf == null) raf = requestAnimationFrame(move); };
    if (!window.__tocSliderWired) {
      window.__tocSliderWired = true;
      window.addEventListener('scroll', schedule, { passive: true });
      window.addEventListener('resize', schedule, { passive: true });
      // NOTE: a document.body-wide class MutationObserver used to re-run move()
      // (a getBoundingClientRect reflow) on EVERY class change Material makes —
      // scrollspy, header autohide, nav state — thrashing layout on every nav
      // click and every scroll frame. Removed; scroll/resize + the timed move()
      // calls below and the per-navigation re-init keep the indicator in sync.
    }
    move();                      // create + position the bar immediately
    setTimeout(move, 400);       // re-position after fonts/layout settle
    setTimeout(move, 1200);      // and once more after instant-nav / late render
  };

  const init = () => {
    enableSectionNumbersOnHome();
    tagSectionEyebrows();
    heroReveal();
    initFadeUp();
    stickyHeader();
    scrollProgress();
    backToTop();
    magneticButtons();
    numberCounters();
    gsapLanding();
    taskFlow();
    tocSlider();
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
  if (typeof document$ !== 'undefined' && document$.subscribe) {
    document$.subscribe(init);
  }
})();
