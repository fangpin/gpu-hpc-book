# 给 MLSys 工程的几个启发

**先让正确性有一个可以信任的锚点。**高性能 kernel 的状态空间太大，如果第一步就同时调度异步 copy、barrier、Tensor Core、multi-CTA 和 epilogue，很难判断错误来自哪里。一个慢但正确的 tiled kernel，是后续每个优化版本的 oracle。

**数据路径比单个 intrinsic 更重要。**GMEM、SMEM、TMEM、register、GMEM 这条链一旦清楚，后续优化就可以逐段替换：线程 copy 换成 TMA，顺序 K-loop 换成 pipeline，单 CTA tile scheduler 换成 persistent scheduling。每次都只改一段，系统复杂度才不会失控。

**同步变量也是程序状态。**`phase_mma ^= 1` 这样的行看起来微小，却承载着 barrier 协议。如果只把它当作模板代码复制，很容易在 refactor 或 loop 重排时引入 silent data corruption。对异步硬件而言，等待语义本身就是算法的一部分。

**Layout 是硬件契约，不是注释。**TIRx 的 layout 写法把“这个 tile 如何映射到 lane、TMEM column 或 thread register row”显式放进程序里。它既影响 lowering 后的指令选择，也影响读写回来的数据是否和数学坐标一致。理解 layout，才真正理解了 tiled GEMM 为什么算对。

---

最后一次更新时间：`2026-08-05 14:01:27 CST`
