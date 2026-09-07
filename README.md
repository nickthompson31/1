# Sunny's Shine Catch

A kids' arcade game for the ShineWater site. One HTML file, zero dependencies,
zero build step, no external assets. Every sprite is drawn on canvas and every
sound is generated in the browser.

**File:** `game/index.html` (~32 KB, single file)

## Play

Catch sun drops, water drops, and rare vitamin D stars. Dodge soda cans and
fizzy candy. Three lives.

| Mechanic | Effect |
|---|---|
| Sun drop | +10 |
| Water drop | +5, fills the Shine Meter fastest |
| Vitamin D star | +50 |
| Combo | +1 multiplier every 5 clean catches, caps at 5x |
| Shine Meter full | **SHINE MODE** — 7s of 2x points plus a magnet that pulls good drops in |
| Level | Every 350 points: faster drops, more hazards |
| Soda / candy | −1 life, combo reset |

Ranks at game over run Sun Sprout → Drop Catcher → Ray Runner → Shine Scout →
Sunbeam Star → Vitamin D Hero → Legend of Light. Best score persists in
`localStorage`.

## Controls

- **Touch / mouse:** drag anywhere to move the bottle
- **Keyboard:** ← → or A / D to move, Space or Enter to start, Esc to pause, M to mute

Portrait-locked 2:3 board that scales to any screen, retina-aware, pauses on tab
blur so it never eats battery in a background tab.

## Hosting it

### Shopify (what we're on)

1. **Settings → Files → Upload** `game/index.html`. Copy the CDN URL Shopify gives you.
2. **Online Store → Pages → Add page**, title it "Sunny's Shine Catch".
3. Switch the content editor to `<>` (HTML) and paste:

```html
<div style="max-width:560px;margin:0 auto;">
  <iframe
    src="PASTE_YOUR_SHOPIFY_FILE_URL_HERE"
    title="Sunny's Shine Catch"
    style="width:100%;aspect-ratio:2/3;border:0;border-radius:22px;display:block;"
    loading="lazy"></iframe>
</div>
```

Shopify serves uploaded files from its CDN, so the game loads fast and costs
nothing in theme weight.

### Anywhere else

Drop `index.html` on any static host — Netlify, Cloudflare Pages, S3, a folder
on the existing server. It's one file with no server requirement. It also works
straight off a USB stick or opened from disk, which is handy for trade shows and
sampling events on a tablet.

## Configuration

Top of the `<script>` block in `game/index.html`:

```js
var CONFIG = {
  storeUrl: "https://shinewater.com",   // where the CTA button points
  ctaText:  "Get ShineWater →",
  lives: 3
};
```

Point `storeUrl` at a discount-code landing page and the game-over screen becomes
a conversion surface instead of a dead end.

## Score hook (email capture / leaderboard)

When embedded in an iframe, the game posts its result to the parent page on every
game over. Listen for it from the host page:

```js
window.addEventListener('message', function (e) {
  if (!e.data || e.data.type !== 'shinewater:gameover') return;
  // e.data → { score, best, level, bestCombo }
  // Show an email form: "Beat 1,500? Get 15% off." Fire an analytics event.
  // Push it into Klaviyo as a custom event.
});
```

That's the hook for a "beat this score for a coupon" promo without touching the
game code.

## Notes

- No tracking, no cookies, no network calls. Safe for a page aimed at kids.
- Audio only starts after the first tap, per browser autoplay rules.
- Facts on the game-over screen are general sunshine/vitamin D statements, not
  product claims — keep it that way if you edit them.
