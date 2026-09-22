"""
utils/validator.py
──────────────────
Deterministic company candidate validation layer.

This is the fix for the core discovery quality bug: raw search results
were being returned directly as "companies" without verifying they are
actually businesses. This module implements every validation rule from
the audit specification.

Validation runs in four passes, each O(1) or O(n) on small fixed sets:

  Pass 1 — TLD rejection         (.gov, .edu)
  Pass 2 — Domain rejection       (directories, aggregators, socials)
  Pass 3 — URL path rejection     (/appointment, /directory, /news, etc.)
  Pass 4 — Title rejection        (Association, Foundation, University, etc.)

A candidate must pass ALL four passes to be kept.

LinkedIn special handling:
  - linkedin.com is NOT rejected at the domain level (unlike Phase 3's
    original behavior).
  - linkedin.com/company/{slug} URLs ARE kept and stored as enrichment data.
  - linkedin.com/search, /feed, /posts, /jobs, /articles ARE rejected
    at the path level.
  - linkedin.com should never appear as a discovered company's primary
    domain — it is enrichment only. See _extract_linkedin_url().
"""

from __future__ import annotations

import re
from urllib.parse import urlparse


# ── Pass 1: TLD rejection ─────────────────────────────────────────────────────

_REJECTED_TLD_PATTERN = re.compile(
    # (^|\.) before "gov|edu|mil" — not just "\." — so the pattern also
    # matches when the TLD label is the very first thing in the domain
    # (e.g. "gov.uk" itself), not only when something precedes it (e.g.
    # "nhs.gov.uk"). Same fix already applied to the "ac" branch below;
    # confirmed real bug: gov.uk — the UK government's own top-level
    # portal — slipped through under a generative-AI/chatbot ICP because
    # the old pattern required a preceding dot that doesn't exist when
    # "gov" is the first label, even though every OTHER real case
    # (nhs.gov.uk, servicesaustralia.gov.au, irs.gov) matched correctly.
    r"(^|\.)(gov|edu|mil)(\.[a-z]{2,3})?$|(^|\.)ac\.[a-z]{2,3}$",
    re.IGNORECASE,
)


def _fails_tld_check(domain: str) -> bool:
    """
    Reject .gov, .edu, .mil TLDs — these are never commercial prospects.

    Also rejects compound country-code forms (.gov.au, .gov.uk, .edu.sg,
    .mil.uk, etc.) — a plain suffix check against ".gov" alone misses
    these, because the domain's actual final label is the ccTLD (.au,
    .uk), not .gov. Confirmed real case: servicesaustralia.gov.au (an
    Australian government agency) was discovered and researched under a
    "local service business" ICP where a government site is never a
    real prospect, despite this check already existing — it simply
    never matched the compound form.
    """
    return bool(_REJECTED_TLD_PATTERN.search(domain))


# ── Pass 2: Domain rejection ──────────────────────────────────────────────────

# Known non-company domains — directories, aggregators, socials, review sites.
# LinkedIn is intentionally NOT here (see module docstring).
_REJECTED_DOMAINS = frozenset({
    # Search engines
    "google.com", "bing.com", "yahoo.com", "duckduckgo.com",
    # Social media (discovery use — enrichment is separate)
    "twitter.com", "x.com", "facebook.com", "instagram.com",
    "youtube.com", "tiktok.com", "pinterest.com", "snapchat.com",
    # Professional networks used as directories — not prospect sites
    "glassdoor.com", "indeed.com", "ziprecruiter.com", "monster.com",
    "careerbuilder.com", "simplyhired.com",
    # Job boards / staffing & recruiting agencies — these are themselves
    # businesses, but they are noise relative to whatever industry an ICP
    # query targets (a "law firms expanding" query returning a staffing
    # agency is off-target, not a lead). Named explicitly per audit
    # findings (e.g. "Robert Half Jobs", "Adzuna").
    "roberthalf.com", "adecco.com", "randstad.com", "manpowergroup.com",
    "kellyservices.com", "michaelpage.com", "hays.com", "adzuna.com",
    "dice.com", "themuse.com", "lever.co", "greenhouse.io",
    "myworkdayjobs.com", "icims.com", "jobvite.com", "smartrecruiters.com",
    "totaljobs.com", "reed.co.uk", "cv-library.co.uk", "careerjet.com",
    "jobstreet.com", "jobsdb.com", "seek.com.au", "seek.co.nz",
    "jooble.org", "talent.com", "snagajob.com", "flexjobs.com",
    # Confirmed in production discovery output: an internship/early-career
    # job board (individual employer postings, same pattern as the other
    # job boards above).
    "wayup.com",
    # Confirmed in production discovery output (see audit evidence):
    # staffing agencies and a niche industry job board.
    "insightglobal.com", "appleone.com", "spacecrew.com",
    # The Built In network — a startup jobs/media platform, exact
    # domains rather than a "builtin" substring keyword: bare "builtin"
    # would false-positive on real built-in cabinetry/closet/furniture
    # businesses (e.g. "custombuiltinteriors.com"), the same class of
    # false-positive risk already documented for bare "chamber" below.
    # Confirmed real case: Built In, Built In Los Angeles, and Built In
    # San Francisco all appeared as three separate "leads" in one
    # 17-company batch under a backend-infrastructure-for-startups ICP —
    # the same non-buyer company counted three times.
    "builtin.com", "builtinla.com", "builtinsf.com", "builtinchicago.org",
    "builtinboston.com", "builtinaustin.com", "builtinseattle.com",
    "builtinnyc.com", "builtincolorado.com",
    # Wellfound (formerly AngelList Talent) — a startup jobs/recruiting
    # platform, not itself a target buyer. Confirmed real case: surfaced
    # under a backend-infrastructure-for-startups ICP.
    "wellfound.com",
    # Fintech Sandbox — a nonprofit providing free data access to fintech
    # startups, not a product company. Same category as the
    # accelerator/incubator substring below, but its own domain doesn't
    # contain either word. Confirmed real case: surfaced under a
    # backend-infrastructure-for-startups ICP.
    "fintechsandbox.org",
    # Review and directory sites
    "yelp.com", "yellowpages.com", "whitepages.com", "bbb.org",
    "trustpilot.com", "g2.com", "capterra.com", "getapp.com",
    # Legal directories / publishers — not the law firms themselves
    "findlaw.com", "avvo.com", "justia.com", "martindale.com",
    "lawyers.com", "superlawyers.com", "nolo.com",
    # Healthcare directories and booking aggregators
    "zocdoc.com", "healthgrades.com", "vitals.com", "webmd.com",
    "ratemds.com", "doximity.com", "practo.com",
    # General aggregators and content platforms
    "wikipedia.org", "wikimedia.org", "reddit.com", "quora.com",
    "medium.com", "substack.com", "wordpress.com", "blogger.com",
    "github.com", "stackoverflow.com", "crunchbase.com",
    "amazon.com", "ebay.com", "etsy.com",
    # Relocation / real-estate content sites named in audit findings
    "migratemate.co", "migratemate.com",
    # Trade-showcase / portfolio content sites named in audit findings
    "officesnapshots.com",
    # News organisations and press-release wire services — not prospects
    "nytimes.com", "wsj.com", "bloomberg.com", "reuters.com",
    "apnews.com", "bbc.com", "cnn.com", "forbes.com",
    "thebusinessdesk.com", "prnewswire.com", "businesswire.com",
    "globenewswire.com", "techcrunch.com", "businessinsider.com",
    "entrepreneur.com", "inc.com", "fastcompany.com",
    # Confirmed in live discovery output under a local-service-business ICP
    # (restaurant/food-industry queries specifically): MSN is a general
    # news/content portal (Wikidata: instance of "web portal") and Daily
    # Mail is a national UK newspaper — same category as nytimes.com/
    # bbc.com/cnn.com above, just not yet on this list. Both TLD variants
    # for Daily Mail included since its own domain is .co.uk but it's
    # commonly indexed/linked under .com redirects too.
    "msn.com", "dailymail.co.uk", "dailymail.com",
    # Confirmed real leak: independent.co.uk (The Independent, a major UK
    # national newspaper) surfaced under an Ireland/New Zealand coffee-shop
    # ICP, tagged "Healthcare" (likely a health-news article got crawled) —
    # domain-level rejection catches it regardless of which section of the
    # site was scraped, unlike a content-based check. Same tier as the
    # nytimes.com/bbc.com/cnn.com entries already above.
    "independent.co.uk",
    # Confirmed real leak: tripadvisor.com — a massive global travel/review
    # directory, not an operating business under any ICP. Surfaced (then
    # excluded only by a crawler block, not by any deliberate check) under
    # the same Ireland/NZ coffee-shop run. Given its scale, this is a
    # significant standing gap independent of that one run.
    "tripadvisor.com",
    # Confirmed in production discovery output: a legal-industry news
    # publisher and a careers/employer-rankings directory site.
    "law.com", "vault.com",
    # Academic publishers/journals — indexed research articles, not companies.
    "plos.org", "ncbi.nlm.nih.gov", "sciencedirect.com", "springer.com",
    "wiley.com", "tandfonline.com", "jstor.org", "researchgate.net",
    "ssrn.com", "papers.ssrn.com", "arxiv.org", "semanticscholar.org",
    # Confirmed real leaks under a logistics/fleet-management ICP:
    # thomasnet.com is a major B2B industrial-supplier directory (same
    # "aggregator, not an operating business" category as tripadvisor.com
    # above, just for industrial suppliers); gc.ca is the Government of
    # Canada's own domain (same category as canada.ca, already covered by
    # _fails_government_site_check for content, but the plain .ca ccTLD
    # means Pass 1's TLD check can't catch it — belt-and-suspenders exact
    # match here since "Government of Canada" is unambiguous); ctfassets.net
    # is Contentful's CDN/asset-hosting domain — never a company at all,
    # just infrastructure a real company's site happens to load images
    # from, occasionally mistaken for the candidate's own site.
    "thomasnet.com", "gc.ca", "ctfassets.net",
    # Confirmed real leak under an auto-repair-shop ICP: autotrader.com —
    # a massive, globally known used/new car classifieds marketplace,
    # same "aggregator platform, not an operating business" category as
    # tripadvisor.com and thomasnet.com above, just for vehicle listings.
    "autotrader.com",
    # Confirmed real leak under a deliberately generic "grow businesses
    # online" ICP: eater.com — a major national food/restaurant
    # journalism publication (Vox Media, now Penske Media/PMX Global),
    # tagged "Food & Beverage" for exactly the reason media outlets keep
    # slipping past industry-token matching: its subject matter overlaps
    # the ICP's target industry vocabulary.
    "eater.com",
    # Confirmed real leak, same run: a cluster of the world's largest
    # global law firms (AmLaw/Global 100 tier, thousands of attorneys
    # each) surfaced under a "law firms" ICP category with employee
    # counts of 51-500 — almost certainly because these firms' sites
    # report a single office's headcount, not firm-wide totals, so the
    # size filter can't catch them. This is a bounded, well-known list
    # (same treatment as the major-newspaper entries above), not a
    # generalized self-description pattern — no confirmed common phrase
    # was found across these firms' homepages to build one safely yet.
    "squirepattonboggs.com", "mcguirewoods.com", "hoganlovells.com",
    "cadwalader.com", "clearygottlieb.com", "seyfarth.com",
    "wilmerhale.com", "mayerbrown.com", "sheppardmullin.com",
    "duanemorris.com",
    # Confirmed real leaks under a daycare/childcare-booking-software ICP:
    # brighthorizons.com — a NYSE-listed (BFAM) global childcare
    # corporation, 32,200+ employees, 1,000+ centers across five
    # countries — same "oversized public company" category as Trek Bikes
    # earlier; mybrightwheel.com — a well-established, widely-used
    # childcare management SaaS platform ("the leading childcare
    # management software" per its own marketing) — this is a direct
    # competitor to the exact product category being demoed, not a
    # prospect, the cleanest vendor/competitor case found so far;
    # niche.com — a massive school/college ratings-and-reviews directory
    # (140M+ reviews), same "aggregator platform" category as
    # tripadvisor.com/thomasnet.com above, just for schools.
    "brighthorizons.com", "mybrightwheel.com", "niche.com",
    # Confirmed real leak under a marketing/real-estate-startup ICP,
    # surfaced via CSV import this time (not web Discovery) — proving
    # this list matters for both lead-discovery paths, not just search
    # results: aeroleads.com, a well-established B2B lead-generation and
    # email-finder SaaS tool. Same "direct competitor to the exact
    # service being demoed" category as mybrightwheel.com above, just
    # for lead-gen/outreach tooling instead of childcare software.
    "aeroleads.com",
    # Startup/company directories — listing pages, not the companies themselves.
    "ycombinator.com", "clutch.co", "goodfirms.co",
    # Major global strategy-consulting firms' own "insights" content —
    # these are thought-leadership articles, not prospect businesses, and
    # the firms themselves are far outside the scale of who this platform
    # is realistically qualifying leads for, regardless of ICP.
    "mckinsey.com", "bain.com", "bcg.com", "deloitte.com", "pwc.com",
    "accenture.com", "ey.com", "kpmg.com",
    # Investor/VC directories and funded-startup lead databases — sell
    # access to lists of *other* companies, never a prospect themselves.
    # Confirmed in production discovery output when ICP queries embedded
    # funding-event language (e.g. "Series B funding").
    "openvc.app", "privateequitylist.com", "cutthrough.com",
    "growthlist.co",
    # Job-listicle aggregators (numeric-count "N+ Role Jobs" pages) not
    # already caught by the "jobs"/"careers" substring keyword below
    # because the domain name itself doesn't contain those words.
    "growthtalent.org",
    # ATS / applicant-tracking platforms — individual employers' job posts
    # live at a per-tenant subdomain (e.g. "flamingoappliance.betterteam.com"),
    # but utils/dedup.root_domain() collapses any 3-label hostname to its
    # last two labels, discarding that subdomain and leaving only the
    # shared platform domain — which is never itself a prospect and has no
    # /about, /services, /careers, /contact pages, so research always fails
    # against it. Rejecting the platform domain outright (same fix pattern
    # as lever.co/greenhouse.io above) is the minimal correct fix here,
    # since root_domain() has no way to know which 3-label hostnames are a
    # real company's own subdomain vs. a multi-tenant platform's.
    "betterteam.com", "vaia.com", "teamworkonline.com",
    # Business-for-sale marketplace — listings of businesses available to
    # purchase, not an operating prospect's own site.
    "bizbuysell.com", "dealstream.com",
    # Regional business news wire (American City Business Journals) — an
    # article ABOUT a company, not the company's own site.
    "bizjournals.com",
    # Startup/company ranking listicle — confirmed noise example from the
    # project's own known-issues list.
    "topstartups.io",
    # Economic development / regional business organization sites —
    # confirmed noise examples from the project's own known-issues list.
    # These are city portals, bilateral trade chambers, and regional
    # economic-development orgs — never an actual prospect company:
    #   houston.org          — Houston's economic development org
    #   businessintexas.com  — Texas state economic development site
    #   gaccmidwest.org      — German American Chamber of Commerce, Midwest
    #   mtec-sc.org          — a South Carolina regional tech/manufacturing
    #                          economic-development council
    "houston.org", "businessintexas.com", "gaccmidwest.org", "mtec-sc.org",
    # Manufacturing/hardtech startup incubator — not itself a manufacturing
    # company; confirmed noise example from the project's own known-issues
    # list (there specifically flagged for a suspicious "1000+" employee
    # count, which is a community/member count, not a headcount).
    "mhub.org",
    # University extension research library — a nonprofit resource/archive
    # about an industry, not a company operating in it. Same "content
    # about a vertical, not a company in it" pattern as bizjournals.com.
    "ruralgrocery.org",
    # Market-research / industry-report publishers — sell reports ABOUT
    # an industry (e.g. "E-commerce Automation Market Size Report"), which
    # makes their own pages keyword-relevant to almost any ICP's search
    # queries without them ever being an operating business in that
    # industry. Same "content about a vertical, not a company in it"
    # pattern as bizjournals.com/ruralgrocery.org above. Confirmed noise
    # from production discovery output under an e-commerce ICP.
    "researchandmarkets.com", "thebusinessresearchcompany.com",
    "ibisworld.com", "databridgemarketresearch.com",
})

# Domain keyword patterns — reject any domain containing these substrings
_REJECTED_DOMAIN_KEYWORDS = (
    "directory", "directories", "listings", "yellowpages",
    "findadoctor", "doctorlist", "healthportal",
    # Job boards / staffing sites not already caught by exact-match above —
    # substring match catches regional/localized variants (e.g. "seek-jobs.co",
    # "xyzstaffing.com") that an exact-domain list can never fully enumerate.
    "jobs", "careers", "recruit", "staffing", "vacanc", "hiring",
    # The iHire job-board network (ihiredental.com, ihirenurses.com,
    # ihireaccounting.com, etc — all the same company, one per vertical).
    # Not caught by "hiring" above since "ihire" doesn't contain that
    # substring. Confirmed exact instance: ihiredental.com surfaced under
    # a dental/vet clinic ICP.
    "ihire",
    # Chambers of commerce and startup accelerators/incubators — same
    # generalization rationale as above: gaccmidwest.org and mhub.org are
    # confirmed exact instances, but there are thousands of local/regional
    # chambers and incubators an exact list will never fully enumerate.
    # NOTE: "chamberofcommerce" (not bare "chamber") — "Chambers" is a
    # common English surname, and surname+trade ("chambersplumbing.com")
    # is an extremely common small-business domain pattern; a bare
    # "chamber" substring was confirmed to false-positive on exactly that.
    "chamberofcommerce", "accelerator", "incubator",
    # Trade/professional associations whose domain names the pattern
    # directly (e.g. american-apartment-owners-association.org) — a
    # confirmed real case that slipped past the content-based
    # _fails_trade_association_check (its title/snippet didn't happen to
    # match those specific phrases), but the domain itself is unambiguous.
    # Hyphenated form, not bare "association", to avoid any risk of
    # matching inside an unrelated compound word.
    "-association",
    # Trade/industry news publications — same rationale as the job-board
    # keywords above: exact-match domains (bizjournals.com) can't cover
    # every niche trade publication ("grocerytradenews.com",
    # "retailnews.com", etc.), and these are articles ABOUT an industry,
    # never a prospect's own site. Accepted tradeoff: a small number of
    # surname-based domains (e.g. "newsome.com") could false-positive here,
    # same class of tradeoff already made for "jobs"/"careers" above.
    "news",
    # US city/municipal government portals — the TLD check (.gov/.edu/.mil,
    # including compound ccTLD forms) doesn't catch these, because US city
    # sites commonly use .us, .org, or a plain .com rather than .gov
    # (confirmed exact instance: cityofnewyork.us). "cityof" is virtually
    # always this exact naming convention regardless of TLD, and — same
    # reasoning as chamberofcommerce/jobs/careers above — an exact-domain
    # list can never enumerate every one of the thousands of US
    # municipalities that follow it.
    "cityof",
    # Meetup — a large, generic community/events platform, structurally
    # never a target buyer under any B2B ICP (same category as the
    # review/directory sites below: g2, capterra, yelp). Confirmed real
    # case: meetup.com surfaced under a startup-backend ICP.
    "meetup",
)



def _fails_domain_check(domain: str) -> bool:
    """Reject known non-company domains and domains with aggregator keywords."""
    if domain in _REJECTED_DOMAINS:
        return True
    for kw in _REJECTED_DOMAIN_KEYWORDS:
        if kw in domain:
            return True
    return False


# ── Search-time noise reduction ───────────────────────────────────────────────
#
# A compact, curated subset of the highest-*volume* noise sources observed
# in production — large, generic job boards and press-release wires that
# show up across almost every ICP query, regardless of industry. Used to
# build a "-site:" exclusion suffix appended to search queries BEFORE they
# reach a provider — see services/discovery_service.py.
#
# Deliberately short: Serper proxies Google search, which has practical
# limits on query complexity, so a long exclusion suffix risks degrading
# result quality more than it helps. Narrower, industry-specific offenders
# named in audit findings (Robert Half, Adzuna, TheBusinessDesk, FindLaw,
# OfficeSnapshots, MigrateMate, etc.) are NOT duplicated here — they are
# already covered by the comprehensive _REJECTED_DOMAINS safety net at
# validation time (Pass 2), which has no query-length constraint. A
# candidate that slips past the search-time exclusion (e.g. from a
# provider that doesn't honor "-site:") is still rejected at validation.
_QUERY_EXCLUSION_DOMAINS = (
    "indeed.com", "glassdoor.com", "ziprecruiter.com", "monster.com",
    "linkedin.com/jobs", "prnewswire.com", "businesswire.com",
)


def negative_site_filters() -> str:
    """
    Build a Google-style '-site:domain' exclusion string from the curated
    noise-domain list above, for appending to outbound search queries.

    Returns a single space-joined string, e.g. "-site:indeed.com
    -site:glassdoor.com ...", ready to concatenate onto a query.
    """
    return " ".join(f"-site:{d}" for d in _QUERY_EXCLUSION_DOMAINS)


# ── Pass 3: URL path rejection ────────────────────────────────────────────────

# Paths that indicate this URL is a resource page, not a company homepage.
# Applied BEFORE domain extraction so we still have path context.
_REJECTED_PATH_SEGMENTS = frozenset({
    "blog", "article", "articles", "news", "press",
    "resource", "resources", "faq", "faqs",
    "appointment", "appointments", "book", "booking", "schedule",
    "directory", "directories", "listing", "listings",
    "search", "find", "locate", "locator",
    "forum", "forums", "community", "discussion",
    "event", "events", "conference",
    "donate", "donation", "volunteer",
    "policy", "policies", "legal", "privacy", "terms",
    "login", "signin", "signup", "register", "account",
    # LinkedIn non-company paths
    "feed", "posts", "jobs", "pulse", "learning",
    "in",   # linkedin.com/in/{person} — personal profiles, not companies
    # Job-posting / career pages — explicitly called out in audit findings
    "career", "careers", "hiring", "vacancy", "vacancies",
    "opportunity", "opportunities", "recruitment",
})

# A small, unambiguous set of keywords worth checking as a SUBSTRING at
# ANY depth in the path — not an exact segment match. Real-world job and
# blog URLs almost never use these words as a clean, isolated path
# segment; they show up fused with other words via hyphens or camelCase,
# which an exact "segment == 'jobs'" check misses entirely. Confirmed by
# production evidence — all of the following slipped past the previous
# exact-match version:
#   jobs.insightglobal.com/find_a_job/california/job-539555
#   appleone.com/CareerSearch/JobDetail/Manufacturing-Operations-Manager...
#   careers.ppg.com/us/en/job/PIIPIIUSJR265292EXTERNALENUS/...
#   spacecrew.com/space-jobs/mg68kpnn-northrop-grumman-...
#   phreesia.com/blog-what-is-patient-intake
#   medozai.com/blogs/manual-patient-intake-cost-ai-healthcare
#   jobs.loves.com/inventory-management-coordinator-retail/job/483783
#   careers.stellantis.com/job/23127456/retail-inventory-management-...
#   careers.sysco.com/en/job/houston/manager-supply-chain-...
#   careers.basspro.com/us/en/job/R253751/Inventory-Analyst-I
# "job" alone (not "jobs") is used as the keyword since substring
# matching means "job" already matches "jobs", "job-539555", "jobdetail",
# etc. Same for "career" matching "careers"/"careersearch". "blog"
# matches "blog", "blogs", and "blog-what-is-...".
# False-positive risk is low and was checked directly against known-good
# results from the same production run: coffeymodica.com's own press-
# release path ("coffey-modica-triples-law-office-footprint-...") and
# duanemorris.com's office page ("/offices/newyork.html") contain none
# of these three keywords and continue to pass.
_DEEP_SCAN_KEYWORDS = ("job", "career", "blog")


def _fails_path_check(url: str) -> bool:
    """
    Reject URLs whose path (or subdomain) indicates they are not a
    company's homepage.

    Checks the FIRST path segment after the domain for the full rejection
    set — so example.com/blog/dental-tips is rejected (first segment =
    'blog') but example.com/dental-services is NOT rejected (no rejected
    segment). Additionally scans the FULL path (any depth, substring
    match) AND the hostname/subdomain for a small high-confidence
    job/career/blog keyword set, since those reliably indicate a job
    posting or blog post wherever they appear — including as a subdomain
    like "jobs.iqvia.com" or "careers.example.com", which a path-only scan
    never sees since the subdomain isn't part of urlparse().path.
    """
    try:
        parsed = urlparse(url if url.startswith("http") else f"https://{url}")
        path = parsed.path.strip("/").lower()
        hostname = (parsed.hostname or "").lower()
    except Exception:
        return False

    if hostname and any(kw in hostname for kw in _DEEP_SCAN_KEYWORDS):
        return True

    if not path:
        return False

    segments = path.split("/")
    if segments[0] in _REJECTED_PATH_SEGMENTS:
        return True
    return any(kw in path for kw in _DEEP_SCAN_KEYWORDS)


# ── Pass 4: Title rejection ───────────────────────────────────────────────────

# Title keywords that indicate a non-commercial entity.
_REJECTED_TITLE_PATTERNS = re.compile(
    r"\b("
    r"association|associations|"
    r"foundation|foundations|"
    r"university|universities|"
    r"college|colleges|"
    r"department|"
    r"government|"
    r"directory|"
    r"resource\s+center|"
    r"board\s+of|"
    r"public\s+health|"
    r"health\s+department|"
    r"county\s+of|"
    r"city\s+of|"
    r"state\s+of|"
    r"national\s+institute|"
    r"centers?\s+for\s+(disease|medicare)|"
    r"wikipedia|"
    r"zocdoc|"
    r"healthgrades|"
    r"webmd|"
    r"yelp"
    r")\b",
    re.IGNORECASE,
)

# Job-posting title patterns — "Jobs at Acme", "Acme is hiring", "Now
# Hiring: ...", "3 Open Positions at Acme". These are job-listing pages
# that slip past the domain/path checks when hosted on an otherwise
# uncategorized job-board domain.
_JOB_TITLE_PATTERNS = re.compile(
    r"\b("
    r"jobs?\s+at|"
    r"careers?\s+at|"
    r"is\s+hiring|"
    r"now\s+hiring|"
    r"we(?:'re|\s+are)\s+hiring|"
    r"job\s+openings?|"
    r"jobs?\s+and\s+openings?|"
    r"open\s+positions?|"
    r"career\s+opportunit\w*|"
    r"apply\s+now|"
    r"vacanc(?:y|ies)"
    r")\b",
    re.IGNORECASE,
)

# Generic content/article/listicle title patterns — blog posts and
# educational articles ("The Ultimate Guide to X", "10 Tips for Y",
# "Patient Intake: Everything You Need to Know") that describe a topic
# rather than name a business, even when hosted on a domain not already
# in the rejected-domain list.
_ARTICLE_TITLE_PATTERNS = re.compile(
    r"(?:"
    r"^\d+\s+(?:tips|ways|reasons|things|steps|mistakes|questions|best)\b|"
    r"\btop\s+\d+\b|"
    r"\bbest\s+\w+(?:\s+\w+){0,4}\s+(?:for|in)\b|"
    r"\bultimate\s+guide\b|"
    r"\bguide\s+to\b|"
    r":\s*a\s+[\w\s]{0,20}\bguide\b|"
    r"\btips\s+for\b|"
    r"\bhow\s+to\b|"
    r"\bwhat\s+is\b|"
    r"\bbenefits\s+of\b|"
    r"\beverything\s+you\s+need\s+to\s+know\b|"
    r"\bchecklist\b|"
    r"\bvs\.?\s+\w+:|"
    r"\bnews\b|"
    r"\bmagazine\b|"
    # Numeric-count listicle/directory titles — "91+ Growth Marketing
    # Manager Jobs in USA 2026", "5,300+ Funded UK Startups 2026". These
    # aren't a single employer's job posting (no "Jobs at X" / "Now
    # Hiring" phrasing for _JOB_TITLE_PATTERNS to catch) — they're an
    # aggregator page enumerating many companies/roles/records at once.
    # A real company's own page is never titled as a running count of
    # itself.
    r"^\d[\d,]*\+?\s+[\w\s]{0,40}\b(?:jobs?|startups?|companies|investors?|leads?|contacts?|firms?)\b"
    r"(?!\s*(?:portfolio\s+company\b|:))|"
    # Investor/lead/startup database and directory platforms — sell
    # access to lists of *other* companies (VCs, funded startups,
    # decision-maker contacts); not themselves a prospect in any ICP.
    r"\b(?:investor|lead|startup|b2b\s+lead)s?\s+database\b|"
    r"\btop\s+(?:saas\s+)?investors?\b|"
    r"^list\s+of\b|"
    r"\bventure\s+capital\s+firms?\s+for\b"
    r")",
    re.IGNORECASE,
)

# Content-based recruitment/staffing-agency detection (title + snippet).
# URL-path filtering (see _fails_path_check) cannot catch a recruitment
# agency's own homepage — there's no "/jobs" or "/careers" segment because
# the agency's entire business IS recruiting, not a specific employer's
# job board. This is the deterministic alternative: match on vocabulary a
# recruitment/staffing agency's own marketing copy reliably uses to
# describe itself, which a genuine target company essentially never uses
# about itself. Kept narrow and phrase-based (not single generic words
# like "recruit" or "hire", which real companies do use when describing
# their own hiring) to avoid false-positiving on companies that are
# simply growing.
#
# Extended after two confirmed misses in production discovery output:
#   lawfirmpeople.com  — "Legal Marketing Recruiters that specialize in
#                         law firm marketing recruitment... Filcro Legal
#                         Staffing recruits legal marketing... personnel"
#   calibrate-strategies.com — "roles are listed on an exclusive basis
#                         with Calibrate and will not be available
#                         through other firms or agencies"
# Neither matched the original phrase set because the original patterns
# required specific word-adjacency ("staffing agency", "staffing firm")
# that real agency copy doesn't always use — "Staffing recruits" and
# "Recruiters that specialize in" are equally clear self-descriptions
# but different word order.
_RECRUITMENT_AGENCY_PATTERNS = re.compile(
    r"\b("
    r"recruitment\s+(?:agency|consultancy|firm|specialists?|team)|"
    r"staffing\s+(?:agency|firm|solutions)|"
    r"executive\s+search|"
    r"talent\s+acquisition\s+partner|"
    r"we\s+work\s+with\s+our\s+candidates|"
    r"our\s+(?:candidates|consultants)\s+(?:to\s+)?(?:provide|have|are)|"
    r"(?:permanent|interim)\s+and\s+(?:permanent|interim)\s+(?:opportunities|roles|positions)|"
    r"recruiters?\s+that\s+specialize|"
    r"staffing\s+recruits|"
    r"listed\s+on\s+an?\s+exclusive\s+basis|"
    r"(?:not\s+be\s+)?available\s+through\s+other\s+(?:firms|agencies|recruiters)"
    r")\b",
    re.IGNORECASE,
)

# General fallback: "staffing" and "recruit*" both appearing anywhere in
# the same title+snippet is a strong combined signal a genuine target
# company's own copy essentially never produces (a law firm describing
# itself doesn't use both words together), even when neither the specific
# phrase patterns above nor a domain-keyword match fires.
_STAFFING_WORD_RE = re.compile(r"\bstaffing\b", re.IGNORECASE)
_RECRUIT_WORD_RE = re.compile(r"\brecruit", re.IGNORECASE)


def _fails_title_check(title: str) -> bool:
    """Reject result titles that indicate non-commercial entities, job
    postings, or generic content/article pages rather than a business."""
    return bool(
        _REJECTED_TITLE_PATTERNS.search(title)
        or _JOB_TITLE_PATTERNS.search(title)
        or _ARTICLE_TITLE_PATTERNS.search(title)
    )


def _fails_recruitment_agency_check(title: str, snippet: str) -> bool:
    """
    Reject recruitment/staffing agencies detected via their own marketing
    language in the title or snippet, since their homepage URL structure
    is indistinguishable from a genuine target company's (see module docs
    on Barclay Simpson — no /jobs path exists because the whole site IS
    the recruiting business).
    """
    combined = f"{title} {snippet}"
    if _RECRUITMENT_AGENCY_PATTERNS.search(combined):
        return True
    # Fallback: "staffing" and "recruit*" both present is a strong combined
    # signal even without matching one of the specific phrases above.
    return bool(_STAFFING_WORD_RE.search(combined) and _RECRUIT_WORD_RE.search(combined))


_TRADE_ASSOCIATION_PATTERNS = re.compile(
    r"\b("
    r"trade\s+association|"
    r"industry\s+association|"
    r"national\s+association\s+of|"
    r"professional\s+association\s+for|"
    r"advocat(?:e|es|ing|acy)\s+for\s+the\s+\w+\s+industry|"
    r"represents?\s+the\s+interests\s+of|"
    r"our\s+member\s+companies|"
    # Confirmed real case: RETTC (rettc.org) surfaced under a
    # backend-infrastructure ICP, tagged "Consulting" — its own site
    # self-describes as "the premier membership organization focused on
    # the intersection of technology and real estate," not "membership
    # organization FOR ___" (the original anchor). Broadened to also
    # catch "focused on"/"dedicated to" and a "premier/leading" lead-in —
    # deliberately still anchored (not a bare "membership organization"
    # match): an early version without any anchor caught a hypothetical
    # gym-management SaaS company's unrelated mention of "membership
    # organization" as a product feature in its own marketing copy.
    r"(?:premier|leading)\s+membership\s+organization|"
    r"membership\s+organization\s+(?:for|focused\s+on|dedicated\s+to)|"
    r"nonprofit\s+association|"
    r"not-for-profit\s+association|"
    r"bar\s+association|"
    r"lawyers?\s+association|"
    r"law\s+society|"
    # Confirmed real case: Finance Montréal (finance-montreal.com)
    # surfaced under the same ICP, tagged "Consulting" — self-describes
    # verbatim as "Québec's Financial Services Cluster" / "financial
    # cluster." An "industry cluster" (regional economic-development
    # concept: a geographic concentration of companies + institutions in
    # one sector, coordinated by a nonprofit body) is a distinct
    # organization type from any of the operating businesses within it.
    r"(?:industry|industrial|financial|services)\s+cluster\b"
    r")\b",
    re.IGNORECASE,
)


def _fails_trade_association_check(title: str, snippet: str) -> bool:
    """
    Reject trade/industry associations detected via their own marketing
    language — same rationale as _fails_recruitment_agency_check: an
    association's homepage URL structure is indistinguishable from a
    genuine target company's, so only its own self-description (a
    nonprofit representing an industry, not an operating business in it)
    reveals the mismatch. Confirmed real case: Canadian Association of
    Movers (mover.net) surfaced under a moving-industry ICP.
    """
    combined = f"{title} {snippet}"
    return bool(_TRADE_ASSOCIATION_PATTERNS.search(combined))


_ECONOMIC_DEVELOPMENT_PATTERNS = re.compile(
    r"\b("
    r"economic\s+development\s+(?:board|council|corporation|authority|"
    r"agency|partnership|alliance|organization|office|team)|"
    r"business\s+development\s+board|"
    r"(?:state|county|regional|metro)\s+economic\s+development|"
    r"site\s+selectors?\s+(?:choose|trust)|"
    r"invest\s+in\s+\w+\s+county|"
    r"why\s+companies\s+(?:choose|relocate\s+to)|"
    r"business\s+location\s+resource|"
    r"relocate\s+or\s+expand\s+within"
    # Added: the exact-domain list (bdb.org, selectflorida.org) and the
    # phrases above it were confirmed via those orgs' about-us pages —
    # but the validator only ever sees homepage title/snippet, and a
    # confirmed real leak (columbusregion.com, surfaced under an
    # AI-automation mid-to-large-org ICP) showed its *homepage* uses
    # different self-description language than its about-us page
    # ("committed to economic growth in the region", "business location
    # resource", "companies... relocate or expand within the Columbus
    # Region") even though the about-us page does contain the original
    # pattern's exact phrase "economic development organization". Added
    # the two homepage-confirmed phrases above; left the more generic
    # "committed to economic growth" out since it's plausible ordinary
    # marketing copy for a real company and would risk false positives.
    r")\b",
    re.IGNORECASE,
)


def _fails_economic_development_check(title: str, snippet: str) -> bool:
    """
    Reject regional/state economic-development organizations detected via
    their own self-description — same rationale and same "homepage is
    indistinguishable from a real company's by URL alone" problem as
    _fails_trade_association_check. The existing exact-domain list
    (houston.org, businessintexas.com, etc.) already covers this category
    but can't generalize — there are as many of these as there are US
    states/counties/regions, each pursuing exactly the "manufacturing
    company expands new facility [location]" and "distribution center
    about us [location]" style discovery queries this project generates,
    since attracting and publicizing those exact companies IS an EDC's
    entire content strategy. Confirmed real cases from live testing:
    "Business Development Board of Palm Beach" (bdb.org) and
    "SelectFlorida" (selectflorida.org) both surfaced under a
    manufacturing/warehouse-employer ICP.
    """
    combined = f"{title} {snippet}"
    return bool(_ECONOMIC_DEVELOPMENT_PATTERNS.search(combined))


# ── LinkedIn enrichment extraction ────────────────────────────────────────────

_LINKEDIN_COMPANY_RE = re.compile(
    r"linkedin\.com/company/([a-zA-Z0-9\-_%]+)/?",
    re.IGNORECASE,
)


def extract_linkedin_url(url: str) -> str | None:
    """
    Extract a valid LinkedIn company page URL from any URL string.

    Returns the canonical URL (linkedin.com/company/{slug}) if the
    URL is a valid company page, None otherwise.

    Valid:   https://linkedin.com/company/acme-dental  → kept
    Invalid: https://linkedin.com/in/john-doe          → None (personal)
    Invalid: https://linkedin.com/jobs/view/123        → None (job listing)
    Invalid: https://linkedin.com/search/results/...   → None (search page)
    """
    match = _LINKEDIN_COMPANY_RE.search(url)
    if match:
        slug = match.group(1).rstrip("/")
        return f"https://www.linkedin.com/company/{slug}"
    return None


# ── Main public interface ─────────────────────────────────────────────────────

class ValidationResult:
    """Outcome of validating a single search result."""

    __slots__ = ("passed", "rejection_reason", "linkedin_url")

    def __init__(
        self,
        passed: bool,
        rejection_reason: str = "",
        linkedin_url: str = "",
    ) -> None:
        self.passed = passed
        self.rejection_reason = rejection_reason
        self.linkedin_url = linkedin_url


_MEDIA_PUBLICATION_PATTERNS = re.compile(
    r"\b("
    r"published\s+by|"
    r"digital\s+publication|"
    r"media\s+outlet|"
    r"editorial\s+team|"
    r"subscribe\s+to\s+our\s+newsletter|"
    r"breaking\s+news|"
    r"latest\s+news\s+and\s+updates|"
    r"(?:travel|business|lifestyle|food)\s+magazine|"
    r"\b\w+[\s-]+(?:chronicle|tribune|gazette|herald|enquirer|advertiser)\b|"
    # "Times" specifically requires a HYPHEN (not a bare space) before
    # it — two place names joined by a hyphen is a distinctive small-
    # town-newspaper naming convention (confirmed real case:
    # Milford-Orange Times), whereas a bare-space "X Times" pattern is
    # too broad and false-positives on real companies (confirmed:
    # "Modern Times Beer Co", "Good Times Restaurant" both matched and
    # would have been wrongly rejected when tested with a space-based
    # version of this rule).
    r"\b\w+-\w+\s+times\b|"
    # Newspaper self-description phrases — a proper-name newspaper (no
    # telltale suffix like "chronicle"/"tribune") needs its own decisive
    # phrase, same rationale as the Condé Nast check below for magazines
    # with no matching suffix. Confirmed real case: The Globe and Mail
    # surfaced under a hospitality ICP; its own about page states
    # verbatim "Canada's foremost news media company" and "national
    # newspaper" — confirmed via direct research, not guessed.
    r"news\s+media\s+company|national\s+newspaper|daily\s+newspaper|"
    # Publisher-name check, not title-by-title: GQ surfaced under a
    # retail ICP and wasn't caught, because the existing check only
    # matched Condé Nast Traveler by coincidental "travel magazine"
    # snippet phrasing — it had no way to recognize the publisher
    # itself. Condé Nast's copyright footer ("© Condé Nast") appears on
    # every page of every property it owns (GQ, Vogue, Wired, Vanity
    # Fair, The New Yorker, Bon Appétit, Architectural Digest, and
    # Condé Nast Traveler itself), so this one line catches the whole
    # family structurally instead of enumerating titles one at a time —
    # the same shape of gap as the generic "[X] Association" pattern
    # added below for trade groups.
    r"cond[ée]\s+nast|"
    # "___Week" trade-publication family — enumerated by name rather than
    # a generic "\w+\s*week" suffix, because "week" alone is far too
    # common in ordinary business copy ("new content every week", "we
    # ship weekly") to use as a pattern the way "chronicle"/"tribune"
    # above safely can. Confirmed real cases: InformationWeek surfaced
    # under a backend-infrastructure ICP (its own About page: "part of
    # Informa TechTarget's media portfolio"); Retail Week surfaced under
    # an earlier ICP run. Adweek/Brandweek/Mediaweek/PRWeek are the same
    # family (confirmed via direct research — all trade publications,
    # several historically related to one another) and included
    # preemptively rather than waiting for each to leak individually.
    r"information\s*week|retail\s*week|ad\s*week|brand\s*week|media\s*week|pr\s*week|"
    # Self-description phrases from a confirmed real case broader than
    # any single outlet: Startup Savant (a startup-advice/content site,
    # not a startup itself) surfaced under a backend-infrastructure ICP.
    # Its own /about-us page states verbatim "resource guide" content
    # produced by an "editorial team" (already covered above), offers a
    # "Media Kit and Advertise page" to would-be advertisers, and
    # discloses that it "earns affiliate commissions" — none of which
    # are phrases a real product/service business would use to describe
    # itself, but all of which are standard, low-false-positive-risk
    # self-descriptions for a media/content/directory site living off
    # advertising and affiliate revenue rather than selling its own
    # product or service.
    r"media\s+kit|sponsored\s+content|affiliate\s+commissions?|"
    r"advertising\s+opportunit(?:y|ies)|advertise\s+with\s+us|"
    r"media\s+brand|media\s+portfolio|"
    # Confirmed real case: birdsadvice.com surfaced as a "Healthcare"
    # lead under a computer-vision ICP (its bird-health vocabulary
    # apparently tripped the deterministic industry classifier), but its
    # own About page is unambiguous: "what started out as a passion
    # project about advice regarding birds, has evolved into a content
    # hub..." — a personal hobby blog, not any kind of operating
    # business. Distinct from the media-company/magazine patterns above:
    # this site never claims to be a publication or outlet, so those
    # phrases don't fire — it self-describes with the more casual,
    # equally decisive language of an amateur content site.
    r"passion\s+project|content\s+hub|hobby\s+blog|"
    # Confirmed real case: advancedmanufacturing.org (SME Media, publisher
    # of "Manufacturing Engineering" and "Smart Manufacturing" magazines)
    # surfaced under an AI-automation ICP tagged "Manufacturing" — a
    # trade publication, not a manufacturer. A BARE "magazine" isn't
    # matched here at all (deliberately — real product companies routinely
    # reference third-party press mentions like "as featured in Forbes
    # magazine," which a bare-word match would wrongly reject). This
    # phrase is scoped to the self-referential "publisher of ... magazine"
    # construction specifically, which only ever describes a site talking
    # about its OWN publication, not someone else's — confirmed verbatim
    # on advancedmanufacturing.org's own homepage: "powered by SME Media,
    # publisher of Manufacturing Engineering and Technology magazine."
    r"publisher\s+of\b.{0,80}\bmagazine\b|"
    # Confirmed real case: BetaKit (betakit.com), a well-known Canadian
    # tech-news outlet, surfaced under a backend-infrastructure ICP
    # (bizarrely tagged "Construction & Home Services" by the industry
    # classifier — this phrase catches it regardless of that mistagging).
    # Its own About page states verbatim: "BetaKit is the publication of
    # record for those who want to understand what's happening in
    # Canadian tech."
    r"publication\s+of\s+record|"
    # Confirmed real case: Startups of London's own homepage states
    # verbatim "Startups of London is a media company that aims to be
    # the voice of the startup ecosystem" and separately advertises
    # itself elsewhere as "London's biggest online startup publication"
    # — a company plainly stating it IS a media company/publication is
    # about as decisive and low-false-positive-risk a signal as this
    # check gets.
    r"(?:is|are)\s+a\s+media\s+company|online\s+publication|"
    # Confirmed real case: foundry-planet.com — a metal-casting industry
    # news/directory portal — surfaced under an AI-automation ICP tagged
    # "Manufacturing." Deliberately chosen as a MORE GENERAL fix than
    # naming foundry-planet.com specifically: its own <title> tag reads
    # "foundry-planet.com | B2B Portal for Technical and Commercial
    # Foundry Management" on every single page including the homepage
    # (which the crawler always visits), and "B2B portal" is a common,
    # genuinely self-descriptive term real trade-media/directory sites
    # use about themselves — not something an actual manufacturer,
    # logistics company, or law firm would ever say about itself. Catching
    # the general term here, rather than only this one outlet's name,
    # should also catch the next several similarly-shaped portals
    # (foundry-planet.com was already the second "trade portal" leak in a
    # row, after industryohio.com in the same run) without needing a new
    # phrase written for each one individually.
    r"b2b\s+portal|b-to-b\s+portal|"
    # Confirmed real case, same run: industryohio.com — a directory/news
    # feed of OTHER Ohio manufacturers' plant openings and expansions —
    # surfaced tagged "Manufacturing." Its own boilerplate category
    # labels, repeated across pages, are distinctive enough to use as a
    # site fingerprint the same way "Just another WordPress site" is used
    # above for WordPress's own default tagline.
    r"expanding\s+industry.{0,10}new\s+industry.{0,10}oem.{0,10}system\s+integration|"
    # Confirmed real case: discoverfoodtrucks.com and streetfoodfinder.com
    # (food-truck "finder"/directory apps) both surfaced under a
    # restaurant-reservation-tool ICP tagged "Food & Beverage" — neither
    # is a food truck operator, both are directories OTHER food trucks
    # list themselves on. "Get your [X] listed" is addressed to other
    # businesses inviting them to join the directory — a real operating
    # business essentially never says this about itself, only a platform
    # recruiting listings does. Confirmed verbatim on discoverfoodtrucks.com:
    # "Get your food truck listed on America's largest directory for FREE!"
    r"get\s+your\s+.{0,30}\s+listed\b|largest\s+directory\b|vendor\s+platform\b"
    r")\b",
    re.IGNORECASE,
)


_PRESS_MENTION_GUARD = re.compile(
    r"(?:featured|seen|mentioned|appeared|covered)\s+(?:in|on|by)\b",
    re.IGNORECASE,
)


_NICHE_TRADE_NEWS_PATTERNS = re.compile(
    r"(?:primary|leading)\s+source\s+for\s+news\s+about|"
    r"industry\s+updates\s+(?:right\s+)?to\s+your\s+(?:mailbox|inbox)|"
    r"curate\s+news\s+from\s+(?:original|primary)\s+sources",
    re.IGNORECASE,
)


def _fails_niche_trade_news_check(title: str, snippet: str) -> bool:
    """
    Reject small, niche-industry trade news outlets — a narrower sibling
    of _fails_trade_association_check and _fails_media_publication_check
    above, aimed at single-topic industry news sites ("X Wire" / "X News"
    style) whose brand name gives no generic hint like "news" or
    "magazine" and are too small/niche to appear on any curated domain
    list. Confirmed real case: dredgewire.com — self-described verbatim
    as "your primary source for news about dredging, ports and
    harbors... DredgeWire delivers industry updates right to your
    mailbox" — surfaced under a logistics/fleet-management ICP purely
    because dredging and shipping share industry vocabulary. The
    "curate news from primary sources" framing is itself a recognizable
    trade-newsletter pattern independent of the specific industry covered.
    """
    combined = f"{title} {snippet}"
    return bool(_NICHE_TRADE_NEWS_PATTERNS.search(combined))


_ACADEMIC_JOURNAL_SELF_DESC_PATTERNS = re.compile(
    r"(?:peer[\s-]?reviewed|open[\s-]?access)\s+(?:scholarly\s+|academic\s+|international\s+)*journal|"
    r"international\s+journal\s+of\s+\w|"
    r"submit(?:ting)?\s+(?:your\s+|a\s+)?research\s+paper|"
    r"publish(?:ing)?\s+(?:your\s+)?research\s+paper",
    re.IGNORECASE,
)


def _fails_academic_journal_self_description_check(title: str, snippet: str) -> bool:
    """
    Reject academic/research journals detected via self-description,
    generalizing the existing exact-domain academic-publisher list
    (plos.org, jstor.org, etc.) above. That list can only ever cover the
    handful of major publishers on it — there are thousands of small and
    "predatory" open-access journals (frequently named "International
    Journal of <field>") that will never all fit a curated list, but
    reliably use the same handful of self-description phrases regardless
    of which one shows up. Confirmed real case: ijirmps.org
    ("International Journal of Innovative Research in Engineering &
    Multidisciplinary Physical Sciences... a bi-monthly multidisciplinary
    international peer reviewed open access scholarly journal... Publish
    Research Paper") surfaced under a logistics/fleet-management ICP,
    tagged "Technology" — presumably because it publishes engineering
    research, which shares enough vocabulary with a tech/logistics ICP
    to score a partial industry match.
    """
    combined = f"{title} {snippet}"
    return bool(_ACADEMIC_JOURNAL_SELF_DESC_PATTERNS.search(combined))


def _fails_media_publication_check(title: str, snippet: str) -> bool:
    r"""
    Reject news outlets, magazines, and digital publications detected via
    self-description — same "homepage URL is indistinguishable from a
    real company's" rationale as the checks above. The existing exact-
    domain approach (bizjournals.com) and path/domain keyword "news"

    can't catch a publication whose brand name doesn't contain the word
    "news" (a magazine's own title, a newspaper's proper name). Confirmed
    real cases from live testing: "Dallas Innovates | Published by Dallas
    Next", "San Francisco Chronicle", "Condé Nast Traveler" all surfaced
    under a hospitality ICP.

    Guard added after this check's own regression test caught it
    false-positiving on ordinary small-business marketing copy: the bare
    "(?:travel|business|lifestyle|food)\s+magazine" branch matches any
    mention of the phrase "food magazine" etc. anywhere in the text, not
    just self-description — e.g. "As featured in a local food magazine,
    come try our specialty roasts" describes a real café being *covered
    by* a magazine, not *being* one, and would have been wrongly
    rejected. _PRESS_MENTION_GUARD skips the whole check when a
    "featured/seen/mentioned/appeared/covered in/on/by" press-mention
    frame is present anywhere in the text — broader than scoping the
    exclusion to just the one magazine branch, but deliberately so: the
    same false-positive shape (a real business citing outside coverage)
    applies equally to "news media company", "national newspaper", etc.
    A genuine magazine/publication would not normally credit itself as
    "featured in" another outlet on its own homepage.
    """
    combined = f"{title} {snippet}"
    if _PRESS_MENTION_GUARD.search(combined):
        return False
    return bool(_MEDIA_PUBLICATION_PATTERNS.search(combined))


_PUBLIC_BROADCASTING_PATTERNS = re.compile(
    r"\b(public\s+broadcasting|public\s+television|public\s+radio|"
    r"\bPBS\b|\bNPR\b)\b",
)


def _fails_public_broadcasting_check(title: str, snippet: str) -> bool:
    """
    Reject public broadcasting stations (TV/radio) — a distinct entity
    type from any operating business, structurally never a real target
    under any ICP. Confirmed real case: pbssocal.org surfaced under a
    hospitality ICP (tagged "Education", 51-200 employees).
    """
    combined = f"{title} {snippet}"
    return bool(_PUBLIC_BROADCASTING_PATTERNS.search(combined))


_LOCAL_BROADCAST_PATTERNS = re.compile(
    r"\b("
    r"(?:nbc|abc|cbs|fox|cw|telemundo|univision)\s+affiliate|"
    r"affiliated\s+with\s+the\s+(?:nbc|abc|cbs|fox|cw)\s+network|"
    r"(?:local\s+news\s+leader|leader\s+in\s+local\s+news)|"
    r"news,?\s+weather\s+and\s+sports\s+coverage"
    r")\b",
    re.IGNORECASE,
)


def _fails_local_broadcast_check(title: str, snippet: str) -> bool:
    """
    Reject commercial local TV/radio station sites — a distinct category
    from _fails_public_broadcasting_check above (that one is scoped to
    PBS/NPR-style *public* broadcasting only). Same "domain is
    indistinguishable from a real company's" problem: a station's call-
    sign or brand domain (nbcrightnow.com, kezj.com) has no structural
    marker, only its own self-description does. Confirmed real case:
    nbcrightnow.com ("NonStop Local Tri-Cities/Yakima") surfaced under a
    local-service-business ICP, self-described verbatim on its own
    LinkedIn/careers copy as "the NBC affiliate in the Tri-Cities and
    Yakima, Washington TV market" and "Eastern Washington's leader in
    local news, weather and sports coverage on television and digital
    platforms." Network-affiliate phrasing and "local news leader" style
    self-description is near-universal boilerplate across US local TV
    station sites of this type, and not language a genuine local-service
    business would plausibly use about itself.
    """
    combined = f"{title} {snippet}"
    return bool(_LOCAL_BROADCAST_PATTERNS.search(combined))


_TOURISM_BOARD_PATTERNS = re.compile(
    r"("
    r"visitors\s+and\s+locals\s+alike|"
    r"your\s+guide\s+to\s+the\s+best\s+of|"
    r"convention\s+(?:and|&)\s+visitors\s+bureau|"
    r"official\s+visitors?\s+guide|"
    r"official\s+tourism\s+website|"
    r"plan\s+your\s+(?:trip|visit)\s+to"
    r")",
    re.IGNORECASE,
)


def _fails_tourism_board_check(title: str, snippet: str) -> bool:
    """
    Reject destination-marketing/tourism-board and visitor-bureau sites —
    same structural problem as the economic-development check above (a
    quasi-government/nonprofit body promoting a place, not an operating
    business), just for tourism rather than business relocation.
    Confirmed real cases: discovernewport.com ("Discover Newport Oregon |
    Vacation, Travel, Things to Do on the Coast") under a location-specific
    local-service-business ICP, and queenstownnz.co.nz — self-titled
    verbatim "Queenstown New Zealand | Official Tourism Website" — under
    an Ireland/New Zealand coffee-shop ICP. Small-town and county-level
    DMOs are unlikely to have a Wikidata page (see utils/wikidata_check.py's
    own documented limitation), so this content-based check is the only
    layer that can catch them.
    """
    combined = f"{title} {snippet}"
    return bool(_TOURISM_BOARD_PATTERNS.search(combined))


_BUSINESS_IMPROVEMENT_DISTRICT_PATTERNS = re.compile(
    r"business\s+improvement\s+district|\bBID\b",
)


def _fails_business_improvement_district_check(title: str, snippet: str) -> bool:
    """
    Reject Business Improvement Districts (BIDs) — a distinct, well-defined
    organizational category (a levy-funded collective representing many
    member businesses in a defined commercial district, common across UK/
    Ireland city centres), structurally never itself an operating business
    under any ICP. Confirmed real case: dublintown.ie ("DublinTown is a
    Business Improvement District (BID) representing around 2,500
    businesses...") surfaced under an Ireland/New Zealand coffee-shop ICP —
    correctly demoted to "Not Recommended" by the qualify LLM, but not
    caught by any deterministic check beforehand. "\bBID\b" is kept
    case-sensitive (no re.IGNORECASE) since lowercase "bid" is an ordinary
    English word (e.g. "submit a bid") that would otherwise false-positive
    constantly; genuine BID self-references consistently capitalize it as
    an acronym.
    """
    combined = f"{title} {snippet}"
    return bool(_BUSINESS_IMPROVEMENT_DISTRICT_PATTERNS.search(combined))


_LIFESTYLE_MAGAZINE_PATTERNS = re.compile(
    r"lifestyle\s+magazine|subscribe\s+to\s+\w+\s+magazine|"
    r"(?:magazine|website)\s+(?:that\s+)?celebrates\s+the\s+art\s+of\s+living",
    re.IGNORECASE,
)


def _fails_lifestyle_magazine_check(title: str, snippet: str) -> bool:
    """
    Reject local/regional lifestyle magazines and multimedia publishing
    platforms — a smaller-scale sibling of _fails_media_publication_check
    above, aimed at city/regional magazines too small to use the more
    generic "media outlet"/"digital publication" self-description phrasing
    that check targets. Confirmed real case: thedenizen.co.nz ("Denizen is
    New Zealand's most influential lifestyle magazine & website"; "the
    Denizen is a multimedia platform that celebrates the art of living
    well") surfaced under an Ireland/New Zealand coffee-shop ICP, tagged
    "Food & Beverage" — its gastronomy coverage reads exactly like a real
    café/restaurant candidate to industry-token matching alone.
    """
    combined = f"{title} {snippet}"
    return bool(_LIFESTYLE_MAGAZINE_PATTERNS.search(combined))


_GOVERNMENT_SITE_PATTERNS = re.compile(
    r"("
    r"official\s+website\s+of\s+the\s+government|"
    r"government\s+of\s+canada|gouvernement\s+du\s+canada|"
    r"official\s+government\s+website|"
    # Confirmed real case: europa.eu (the European Union's own portal)
    # surfaced under a computer-vision ICP. Same structural gap as
    # Canada.ca below — .eu is an ordinary ccTLD indistinguishable from
    # any EU-based business domain, but the EU's own pages repeat this
    # exact phrase verbatim across every institution's site (European
    # Commission, European Environment Agency, etc.), confirmed via
    # direct research.
    r"official\s+website\s+of\s+the\s+european\s+union|"
    r"\.gov\.\w{2}\b|"  # a .gov.xx TLD mentioned in body text even if the
                        # URL itself normalized differently before reaching
                        # this check
    # Financial regulators — a distinct entity type from a bank/lender
    # (the Banking & Financial Services category above), with domains
    # that don't reliably use a government TLD (fca.org.uk uses .org.uk,
    # not .gov.uk). Confirmed via direct research, not guessed: the FCA
    # is universally self-described with these exact phrases across
    # its own regulator profile, Wikipedia, and third-party summaries.
    # Confirmed real case: fca.org.uk surfaced under a generative-AI ICP.
    r"financial\s+regulator|independent\s+public\s+body|"
    r"the\s+regulator\s+for|"
    # Regional/state economic-development organizations — nonprofit or
    # quasi-governmental bodies whose entire purpose is recruiting OTHER
    # companies to relocate/expand in their region, not operating
    # businesses themselves. Confirmed real case: Ohio Southeast Economic
    # Development (ohiose.com) surfaced under an AI-automation ICP — its
    # own site and press describe it verbatim as a "JobsOhio Network
    # Partner" providing "economic development work," a nonprofit whose
    # job is attracting businesses TO the region, not itself being one.
    #
    # Deliberately anchored to a self-referential "is/serves as the ...
    # [organization]" shape rather than a bare "economic development
    # corporation/authority" noun phrase — an un-anchored version matched
    # a real AI-automation vendor's own marketing copy ("we help economic
    # development corporations modernize...", describing its CLIENTS),
    # the same plural/third-party-mention trap as the "Fortune 500
    # companies" case elsewhere in this file.
    r"(?:is|serves\s+as)\s+(?:the|a)\s+(?:\w+\s+){0,3}"
    r"(?:economic\s+development\s+(?:corporation|authority|organization|partner)|"
    r"regional\s+development\s+(?:organization|authority))|"
    r"jobsohio\s+network\s+partner|"
    r"nonprofit\s+.{0,40}economic\s+development"
    r")",
    re.IGNORECASE,
)


def _fails_government_site_check(title: str, snippet: str) -> bool:
    """
    Reject government sites whose TLD doesn't flag them via the Pass 1
    .gov/.edu/.mil check — same structural gap as US city/municipal
    portals (see the .us/.org note on that pass), but for national
    government sites that use their country's plain ccTLD instead of a
    dedicated government TLD. Canada is the confirmed real case:
    canada.ca surfaced under an AI-automation-for-mid-to-large-orgs ICP.
    Its .ca TLD is indistinguishable from any ordinary Canadian business
    domain, but the site's own content is unambiguous: the homepage
    title is "Home - Canada.ca" and the body says, verbatim, "The
    official website of the Government of Canada."
    """
    combined = f"{title} {snippet}"
    return bool(_GOVERNMENT_SITE_PATTERNS.search(combined))


_HOTEL_CONSORTIUM_PATTERNS = re.compile(
    r"\b("
    r"hotel\s+consortium|"
    r"collection\s+of\s+(?:independent\s+)?hotels|"
    r"hotel\s+(?:representation|marketing)\s+(?:company|consortium|group)|"
    r"global\s+hotel\s+(?:brand|alliance)|"
    r"world'?s\s+leading\s+hotels"
    r")\b",
    re.IGNORECASE,
)


def _fails_hotel_consortium_check(title: str, snippet: str) -> bool:
    """
    Reject hotel marketing consortiums / booking alliances (a body that
    represents and promotes many independently-owned hotels, not an
    operating hotel itself) — same rationale as the trade-association
    check, scoped to hospitality. Confirmed real case: SLH (Small Luxury
    Hotels of the World) surfaced under a boutique-hotel ICP, tagged
    "Retail".
    """
    combined = f"{title} {snippet}"
    return bool(_HOTEL_CONSORTIUM_PATTERNS.search(combined))


_BUILT_IN_TITLE_PATTERN = re.compile(r"^built\s+in\b", re.IGNORECASE)


def _fails_built_in_network_check(title: str) -> bool:
    """
    Reject the Built In network (a startup jobs/media platform) by its
    distinctive title prefix rather than enumerating domains. Last
    session's fix added exact domains (builtin.com, builtinla.com,
    builtinsf.com, builtinchicago.org, builtinboston.com, etc.)
    deliberately avoiding a bare "builtin" domain-substring match, which
    would false-positive on real built-in cabinetry/furniture businesses
    — but that exact-domain list just proved insufficient: confirmed
    real case, builtinlondon.uk (a domain never seen before, from a
    country the network apparently expanded into) surfaced under a
    generative-AI ICP. A domain list can never be enumerated ahead of
    this company's own expansion. Every confirmed instance's title
    starts with the exact two-word phrase "Built In" — the required
    word-boundary space plus title-start anchor keeps false-positive
    risk low despite "built in" being an ordinary English phrase (a real
    company's title starting with those two words specifically, as
    opposed to the phrase appearing mid-sentence elsewhere, is
    vanishingly rare).
    """
    return bool(_BUILT_IN_TITLE_PATTERN.search(title))


_EVENT_TITLE_PATTERN = re.compile(
    r"\b(?:week|summit|conference|expo|symposium|forum)\s+20\d{2}\b|"
    r"\b20\d{2}\s+(?:week|summit|conference|expo|symposium|forum)\b",
    re.IGNORECASE,
)


def _fails_event_check(title: str) -> bool:
    """
    Reject any result whose TITLE names an event or conference rather
    than a company — a decisive, generalizable pattern (an adjacent
    4-digit year next to "Week"/"Summit"/"Conference"/"Expo"/
    "Symposium"/"Forum") rather than a domain enumeration, since events
    are named individually every year and a domain list could never
    keep up with them the way it can with a stable entity like a job
    board. Real operating companies essentially never put a year in
    their own name this way, which keeps false-positive risk low despite
    "Summit"/"Forum" alone being plausible real brand words (e.g.
    "Summit Bank") — it's the year pairing that makes this decisive.
    Confirmed real case: "Boston Fintech Week 2026" appeared as a scored
    lead under a backend-infrastructure-for-startups ICP — a conference,
    not a company.
    """
    return bool(_EVENT_TITLE_PATTERN.search(title))


_GENERIC_ASSOCIATION_TITLE_PATTERN = re.compile(
    r"\b(association(?!\s+with)|council|magazine|directory)\b", re.IGNORECASE
)


def _fails_generic_association_check(title: str) -> bool:
    """
    Reject any result whose TITLE (deliberately not the snippet, to avoid
    noise from an unrelated "in association with" mention elsewhere, or
    a real company's about-us page mentioning "as featured in X
    magazine") contains the word "association", "council", "magazine",
    or "directory" as a business-name pattern. Real operating companies
    essentially never name themselves "___ Association", "___ Council",
    "___ Magazine", or "___ Directory" — that naming convention is
    reserved for trade groups, nonprofits, membership bodies,
    publications, and listing sites, and the industry-specific phrase
    lists elsewhere in this module can't cover every one of them. The
    negative lookahead excludes "in association with X" phrasing, which
    is a different, legitimate use of the word.
    Confirmed real cases: floridaautobodyassociation.com surfaced under
    an auto-repair ICP; Retail Council of New York State surfaced under
    a retail/e-commerce ICP; LODGING Magazine surfaced under a
    hospitality ICP; "USA University College Directory" surfaced under
    a generative-AI ICP (its domain, stateuniversity.com, didn't trigger
    the domain-keyword "directory" check, which only matches the domain
    string, not the page title) — none caught by the more specific
    patterns.
    """
    return bool(_GENERIC_ASSOCIATION_TITLE_PATTERN.search(title))


def validate_candidate(url: str, title: str, domain: str, snippet: str = "") -> ValidationResult:
    """
    Run all five validation passes against a single search result.

    Args:
        url:     The full source URL from the search provider.
        title:   The search result page title.
        domain:  Already-normalized root domain (from utils/dedup.extract_domain).
        snippet: The search result snippet/description text. Optional —
                 defaults to "" so existing callers that only have url/
                 title/domain keep working unchanged. Only used by Pass 5
                 (content-based recruitment-agency detection), since a
                 recruitment agency's own homepage has no distinguishing
                 URL path (see _fails_recruitment_agency_check).

    Returns:
        ValidationResult with .passed = True if all checks pass.
        If .passed = False, .rejection_reason explains which check failed.

    LinkedIn special case:
        linkedin.com/company/{slug} URLs are kept as enrichment data
        (stored in .linkedin_url) and are NOT treated as company domains.
        The candidate's primary domain must come from elsewhere.
    """
    # LinkedIn company pages — extract for enrichment but don't validate
    # the linkedin.com domain as the company's own site.
    linkedin_url = extract_linkedin_url(url)
    if domain == "linkedin.com" or domain.endswith(".linkedin.com"):
        if linkedin_url:
            # Valid company LinkedIn page — useful as enrichment, but this
            # result should not set the company's primary domain as linkedin.com.
            # Return as passed=False so it doesn't become a discovered_company
            # row, but the caller can still extract the linkedin_url for use.
            return ValidationResult(
                passed=False,
                rejection_reason="linkedin_enrichment_only",
                linkedin_url=linkedin_url,
            )
        return ValidationResult(
            passed=False,
            rejection_reason="linkedin_non_company_page",
        )

    # Pass 1: TLD
    if _fails_tld_check(domain):
        return ValidationResult(passed=False, rejection_reason=f"rejected_tld:{domain.split('.')[-1]}")

    # Pass 2: Domain
    if _fails_domain_check(domain):
        return ValidationResult(passed=False, rejection_reason=f"rejected_domain:{domain}")

    # Pass 3: URL path
    if _fails_path_check(url):
        try:
            first_seg = urlparse(url).path.strip("/").split("/")[0]
        except Exception:
            first_seg = "unknown"
        return ValidationResult(passed=False, rejection_reason=f"rejected_path:/{first_seg}")

    # Pass 4: Title
    if _fails_title_check(title):
        return ValidationResult(passed=False, rejection_reason=f"rejected_title:{title[:40]}")

    # Pass 5: Content (title + snippet) — catches recruitment/staffing
    # agencies whose own homepage has no distinguishing URL path (Pass 3
    # can't see it) or title keyword (Pass 4 can't see it either).
    if _fails_recruitment_agency_check(title, snippet):
        return ValidationResult(passed=False, rejection_reason="rejected_content:recruitment_agency")

    # Pass 6: Content — trade/industry associations (same rationale as
    # Pass 5, different category: a nonprofit representing an industry,
    # not an operating business in it).
    if _fails_trade_association_check(title, snippet):
        return ValidationResult(passed=False, rejection_reason="rejected_content:trade_association")

    # Pass 7: Content — regional/state economic-development organizations
    # (same rationale as Pass 6, different category: a public body that
    # publicizes companies choosing its region, not an operating business
    # itself). Confirmed real cases: bdb.org, selectflorida.org.
    if _fails_economic_development_check(title, snippet):
        return ValidationResult(passed=False, rejection_reason="rejected_content:economic_development")

    # Pass 8: Content — news outlets, magazines, digital publications.
    # Confirmed real cases: dallasinnovates.com, sfchronicle.com,
    # cntraveler.com.
    if _fails_media_publication_check(title, snippet):
        return ValidationResult(passed=False, rejection_reason="rejected_content:media_publication")

    # Pass 9: Content — public broadcasting stations. Confirmed real
    # case: pbssocal.org.
    if _fails_public_broadcasting_check(title, snippet):
        return ValidationResult(passed=False, rejection_reason="rejected_content:public_broadcasting")

    # Pass 9b: Content — commercial local TV/radio station sites (network
    # affiliates), distinct from Pass 9's public-broadcasting scope.
    # Confirmed real case: nbcrightnow.com.
    if _fails_local_broadcast_check(title, snippet):
        return ValidationResult(passed=False, rejection_reason="rejected_content:local_broadcast")

    # Pass 10: Content — government sites whose TLD isn't .gov/.edu/.mil
    # (Pass 1 can't see it). Confirmed real case: canada.ca — Canada's
    # national government portal uses the plain .ca ccTLD.
    if _fails_government_site_check(title, snippet):
        return ValidationResult(passed=False, rejection_reason="rejected_content:government_site")

    # Pass 10b: Content — destination-marketing/tourism-board and
    # visitor-bureau sites. Confirmed real cases: discovernewport.com,
    # queenstownnz.co.nz.
    if _fails_tourism_board_check(title, snippet):
        return ValidationResult(passed=False, rejection_reason="rejected_content:tourism_board")

    # Pass 10c: Content — Business Improvement Districts (BIDs). Confirmed
    # real case: dublintown.ie.
    if _fails_business_improvement_district_check(title, snippet):
        return ValidationResult(passed=False, rejection_reason="rejected_content:business_improvement_district")

    # Pass 10d: Content — local/regional lifestyle magazines and
    # multimedia publishing platforms. Confirmed real case: thedenizen.co.nz.
    if _fails_lifestyle_magazine_check(title, snippet):
        return ValidationResult(passed=False, rejection_reason="rejected_content:lifestyle_magazine")

    # Pass 10e: Content — small niche-industry trade news outlets ("X
    # Wire"/"X News" style) too small for any curated domain list.
    # Confirmed real case: dredgewire.com.
    if _fails_niche_trade_news_check(title, snippet):
        return ValidationResult(passed=False, rejection_reason="rejected_content:niche_trade_news")

    # Pass 10f: Content — academic/research journals detected via
    # self-description, generalizing the exact-domain academic-publisher
    # list above to catch journals not on it. Confirmed real case:
    # ijirmps.org.
    if _fails_academic_journal_self_description_check(title, snippet):
        return ValidationResult(passed=False, rejection_reason="rejected_content:academic_journal")


    # Pass 11: Content — hotel marketing consortiums/alliances (same
    # rationale as trade associations, scoped to hospitality). Confirmed
    # real case: SLH (Small Luxury Hotels of the World).
    if _fails_hotel_consortium_check(title, snippet):
        return ValidationResult(passed=False, rejection_reason="rejected_content:hotel_consortium")

    # Pass 12: Title — generic "[X] Association" naming pattern, catching
    # any trade/membership body the specific phrase lists above miss.
    # Confirmed real case: floridaautobodyassociation.com.
    if _fails_generic_association_check(title):
        return ValidationResult(passed=False, rejection_reason="rejected_content:generic_association")

    # Pass 13: Title — events/conferences, structurally never a company.
    # Confirmed real case: "Boston Fintech Week 2026".
    if _fails_event_check(title):
        return ValidationResult(passed=False, rejection_reason="rejected_content:event")

    # Pass 14: Title — the Built In network, by title prefix rather than
    # domain enumeration (see _fails_built_in_network_check docstring
    # for why the domain-list approach proved insufficient).
    if _fails_built_in_network_check(title):
        return ValidationResult(passed=False, rejection_reason="rejected_content:built_in_network")

    return ValidationResult(passed=True, linkedin_url=linkedin_url or "")