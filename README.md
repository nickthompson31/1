# ShineWater Kids Games

Two browser arcade games for the ShineWater site. Each is one HTML file with
zero dependencies, zero build step, and no external assets — sprites are drawn
on canvas, sounds are synthesized in the browser. Drop either on any static host
or upload it to Shopify Files and iframe it.

| Game | File | Genre | Size |
|---|---|---|---|
| **Sugar Kong — Rescue the Sun** | `games/sugar-kong/index.html` | Donkey Kong-style climber | ~39 KB |
| **Sunny's Shine Catch** | `games/shine-catch/index.html` | Catch-and-dodge | ~32 KB |

Both share the same brand cast: **Sunny** (the sun), the ShineWater bottle, sugar
and soda as the villains, and general vitamin-D facts on the result screens.

---

## Sugar Kong — Rescue the Sun

Sugar Kong hauled Sunny to the top of the tower. You're a kid in a ShineWater
shirt. Climb six floors of girders and ladders, dodge or jump the rolling sugar
barrels, and reach the cage.

| Mechanic | Effect |
|---|---|
| Jump over a barrel | +100 |
| Hammer pickup (2 per floor) | 6s of smashing barrels, +300 each — but no ladders while you hold it |
| ShineWater bottle pickup (3 per floor) | +500 |
| Reach Sunny | Floor cleared, remaining Bonus added to score |
| Bonus | Starts at 5,000 per floor, ticks down 100 every 0.9s |
| Next floor | Faster barrels, quicker throws; from floor 2, purple "wild" barrels always take ladders |
| Barrel hit / fall | −1 life, respawn at the bottom. 3 lives |

Barrels roll toward the open end of each girder (marked with a gold cap), drop
to the next one, and randomly take ladders down. Kong winds up before each throw
so you can read it.

**Controls**
- Keyboard: ← → or A/D walk, ↑ ↓ or W/S climb, Space / Z jump, Esc pause, M mute
- Touch: on-screen d-pad + JUMP button (shown only on touch devices)

Ranks: Ground Floor Rookie → Ladder Legs → Barrel Dodger → Hammer Hand →
Tower Climber → Kong Wrangler → Sun Rescuer.

## Sunny's Shine Catch

Move the ShineWater bottle to catch sun drops (+10), water drops (+5), and rare
vitamin D stars (+50). Dodge soda and candy. Combo multiplier caps at 5x. Fill
the Shine Meter for 7s of **SHINE MODE** (2x points + magnet). Three lives.

**Controls:** drag anywhere, or ← → / A D. Space starts, Esc pauses, M mutes.

---

## Hosting

### Shopify (what we're on)

1. **Settings → Files → Upload** the game's `index.html`. Copy the CDN URL.
2. **Online Store → Pages → Add page**, name it after the game.
3. Switch the editor to `<>` (HTML) and paste:

```html
<div style="max-width:560px;margin:0 auto;">
  <iframe
    src="PASTE_SHOPIFY_FILE_URL_HERE"
    title="Sugar Kong — Rescue the Sun"
    style="width:100%;aspect-ratio:0.62;border:0;border-radius:22px;display:block;"
    loading="lazy"></iframe>
</div>
```

Use `aspect-ratio:2/3` for Shine Catch, `aspect-ratio:0.62` for Sugar Kong (its
touch control strip needs the extra height on phones).

### Anywhere else

Any static host — Netlify, Cloudflare Pages, S3, a folder on the current server.
They also run straight off a tablet with no internet, which is the play for
sampling events and trade show booths.

## Configuration

Top of the `<script>` block in each file:

```js
var CONFIG = {
  storeUrl: "https://shinewater.com",   // where the CTA on the game-over card points
  ctaText:  "Get ShineWater →",
  lives: 3
};
```

Point `storeUrl` at a discount-code landing page and the game-over screen becomes
a conversion surface.

## Score hook (email capture / leaderboard / Klaviyo)

Both games post their result to the parent page on game over:

```js
window.addEventListener('message', function (e) {
  if (!e.data || e.data.type !== 'shinewater:gameover') return;
  // Sugar Kong:   { game:'sugar-kong', score, best, floor }
  // Shine Catch:  { score, best, level, bestCombo }
  // Gate an email form on score. Fire a Klaviyo event. Hand out a coupon.
});
```

## Notes

- No tracking, no cookies, no network calls. Safe for a page aimed at kids.
- Best score and mute preference live in `localStorage` only.
- Audio starts after the first tap, per browser autoplay rules.
- The facts on the result screens are general sunshine/vitamin D statements, not
  product claims. Keep it that way when editing.
- `window.__sugarKong()` is a read-only peek used by the automated playtest.
  Harmless in production; delete it if you'd rather not ship it.
