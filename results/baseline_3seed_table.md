### RoboMME baseline reproduction — MME-VLA `perceptual-framesamp-modul`, checkpoint 79999

The paper averages 9 runs (3 checkpoints x 3 seeds); only checkpoint 79999 was released,
so ours is 3 seeds on that one checkpoint. 50 episodes per task, 800 per seed.
`z` compares the two means against their combined spread.

| Task | Suite | s7 | s17 | s27 | **Ours** | sd | **Paper** | sd | Δ | z |
|---|---|---|---|---|---|---|---|---|---|---|
| BinFill | Coun | 46 | 44 | 40 | **43.3** | 3.1 | 39.56 | 5.27 | +3.8 | +0.62 |
| PickXtimes | Coun | 94 | 92 | 92 | **92.7** | 1.2 | 87.33 | 2.45 | +5.3 | +1.97 |
| SwingXtimes | Coun | 94 | 84 | 90 | **89.3** | 5.0 | 92.00 | 2.24 | -2.7 | -0.48 |
| StopCube | Coun | 44 | 50 | 50 | **48.0** | 3.5 | 42.00 | 11.36 | +6.0 | +0.51 |
| VideoUmsk | Perm | 34 | 28 | 28 | **30.0** | 3.5 | 32.67 | 3.16 | -2.7 | -0.57 |
| ButtonUmsk | Perm | 26 | 18 | 14 | **19.3** | 6.1 | 25.11 | 3.18 | -5.8 | -0.84 |
| VideoUmskS | Perm | 28 | 26 | 20 | **24.7** | 4.2 | 24.44 | 6.06 | +0.2 | +0.03 |
| ButtonUmskS | Perm | 14 | 22 | 20 | **18.7** | 4.2 | 18.22 | 3.80 | +0.4 | +0.08 |
| PickHighL | Refe | 20 | 22 | 22 | **21.3** | 1.2 | 22.89 | 3.89 | -1.6 | -0.38 |
| VideoRepick | Refe | 28 | 24 | 30 | **27.3** | 3.1 | 30.44 | 5.81 | -3.1 | -0.47 |
| VideoPlcBtn | Refe | 60 | 50 | 56 | **55.3** | 5.0 | 60.00 | 4.00 | -4.7 | -0.73 |
| VideoPlcOrd | Refe | 42 | 42 | 40 | **41.3** | 1.2 | 32.00 | 3.87 | +9.3 | +2.31 ⚠️ |
| MoveCube | Imit | 86 | 80 | 84 | **83.3** | 3.1 | 77.78 | 3.80 | +5.6 | +1.14 |
| InsertPeg | Imit | 8 | 6 | 4 | **6.0** | 2.0 | 7.56 | 3.57 | -1.6 | -0.38 |
| PatternLock | Imit | 52 | 64 | 60 | **58.7** | 6.1 | 53.56 | 4.56 | +5.1 | +0.67 |
| RouteStick | Imit | 68 | 66 | 70 | **68.0** | 2.0 | 66.67 | 4.12 | +1.3 | +0.29 |
| **AVG** | | **46.5** | **44.9** | **45.0** | **45.5** | 0.9 | **44.51** | — | +0.9 | — |

| Suite | Ours | Paper |
|---|---|---|
| Counting (n=4) | 68.3 | 65.2 |
| Permanence (n=4) | 23.2 | 25.1 |
| Reference (n=4) | 36.3 | 36.3 |
| Imitation (n=4) | 54.0 | 51.4 |

**AVG 45.5 ± 0.9 vs the paper's 44.51.** 1 of 16 tasks differ by |z| ≥ 2 (marked ⚠️).
