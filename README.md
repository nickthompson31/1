# ShineWater Kids Games

Three browser arcade games for the ShineWater site. Each is one HTML file with
zero dependencies, zero build step, and no external assets — sprites are drawn
on canvas, sounds are synthesized in the browser. Drop either on any static host
or upload it to Shopify Files and iframe it.

| Game | File | Genre | Size |
|---|---|---|---|
| **Shine Run** | `games/shine-run/index.html` | Endless rooftop runner (landscape) | ~32 KB |
| **Sugar Kong — Rescue the Sun** | `games/sugar-kong/index.html` | Donkey Kong-style climber | ~39 KB |
| **Sunny's Shine Catch** | `games/shine-catch/index.html` | Catch-and-dodge | ~32 KB |

Both share the same brand cast: **Sunny** (the sun), the ShineWater bottle, sugar
and soda as the villains, and general vitamin-D facts on the result screens.

---

## Shine Run

Auto-scrolling rooftop dash. The kid runs; you jump and slide. Speed climbs
from a jog to a sprint over the first couple of minutes. The sky runs a full
day — sunrise, noon, sunset, night with a moon and lit windows — every ~1,200m,
and Sunny arcs across it.

| Mechanic | Effect |
|---|---|
| Soda can / sugar stack | Jump it. Hold jump for a higher arc, tap for a short hop |
| Candy-floss cloud | Slide under it (standing hitbox is 56px, sliding is 28px) |
| Rooftop gap | Jump it. Appears after 50m, widens with speed |
| Slide in mid-air | Fast-fall — drop out of a jump early |
| ShineWater bottle | +50, come in jump-shaped arcs of five |
| Sun ray | Collect 5 for **Shine Mode**: 6s invincible, 1.25x speed, 2x score, magnet on pickups, obstacles smash for +100 |
| Score | Distance x 0.1 + pickups |

**Controls**
- Keyboard: Space / ↑ / W jump, ↓ / Shift / S slide, Esc pause, M mute. Mouse click also jumps.
- Touch: the screen splits into two zones — hold **left** to slide, tap **right** to jump. Labels show only on touch devices.

Ranks: Rooftop Rookie → Puddle Jumper → Ledge Leaper → Skyline Sprinter →
Sunrise Chaser → Shine Runner → Legend of Light.

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

Aspect ratios per game: Shine Run `16/9` (bump `max-width` to 1100px), Shine
Catch `2/3`, Sugar Kong `0.62` (its touch control strip needs the extra height
on phones).

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

All three games post their result to the parent page on game over:

```js
window.addEventListener('message', function (e) {
  if (!e.data || e.data.type !== 'shinewater:gameover') return;
  // Shine Run:    { game:'shine-run', score, best, distance, bottles }
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
- `window.__sugarKong()` / `window.__shineRun()` are read-only peeks used by the
  automated playtests. Harmless in production; delete them if you'd rather not
  ship them.
