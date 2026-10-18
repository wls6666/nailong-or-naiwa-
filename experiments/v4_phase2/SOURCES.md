# Cartoon source attribution

Frames/contact sheets are derived from the following publicly downloadable open movies. Metadata, license URL, original URL, video hash, frame timestamp, pixel/file/perceptual hashes are in `data/v4_phase2/cartoon_manifest.json`. Downloaded videos and individual frames are not committed. Contact sheets preserve film filenames as source identifiers; this table supplies attribution. No BY-ND source was used.

| Work / credit | Public distribution and license | Use |
|---|---|---|
| Big Buck Bunny — Blender Foundation / Peach team | https://archive.org/details/BigBuckBunny_328 — https://creativecommons.org/licenses/by/3.0/us/ | training |
| Caminandes 3: Llamigos — Blender Foundation / Caminandes team | https://archive.org/details/CaminandesLlamigos — https://creativecommons.org/licenses/by/3.0/ | training |
| Cosmos Laundromat: First Cycle — Blender Foundation / Gooseberry team | https://archive.org/details/CosmosLaundromatFirstCycle — https://creativecommons.org/licenses/by/4.0/ | training |
| Sintel — Blender Foundation / Durian team | https://archive.org/details/Sintel — https://creativecommons.org/licenses/by/3.0/ | validation only |
| Sprite Fright — Blender Studio | https://archive.org/details/sprite-fright — https://creativecommons.org/licenses/by/4.0/ | final test only |
| Wing It! — Blender Studio / Rik Schutte | https://archive.org/details/wing_it — https://creativecommons.org/licenses/by/4.0/ | final test only |

Changes: deterministic frame extraction, resize to480px width and JPEG encoding, perceptual deduplication, visually reviewed content filtering, thumbnails/contact-sheet composition. Character identities and source movies remain credited to their respective creators. License statements are retained from public archive item metadata; the project does not claim to own these characters. All six works share the Blender open-movie ecosystem, which limits source/style diversity despite distinct held-out casts and films.
