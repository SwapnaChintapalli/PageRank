//class PageRank {
//
//}
import org.apache.spark.SparkContext

object pagerank1 {

  def main(args: Array[String]) {
    if (args.length != 3) {
      println("Usage: PageRank inputDir outputDir nIterations")
    }
    System.setProperty("hadoop.home.dir", "G:\\IntelliJ\\PageRankFinal\\execfile")
    val inputPath = if (args.length > 0) args(0) else "G:\\IntelliJ\\PageRankFinal\\input\\87687717_T_ONTIME_REPORTING\\87687717_T_ONTIME_REPORTING.csv"

    val outputDir = if (args.length > 1) args(1) else "G:\\IntelliJ\\PageRankFinal\\output"

    val iter: Int = if (args.length > 2) args(2).toInt else 50

    val alpha = 0.15

    val master = "local"
    val sc = new SparkContext(master, "PageRank", System.getenv("SPARK_HOME"))
//    val sc = new SparkContext(new SparkConf().setAppName("Spark Count"))
    sc.setLogLevel("ERROR")


    val data = sc.textFile(inputPath)
    val header = data.first()
    val links = data.filter(row => row != header).map(s => {
      val pairs = s.split(",")
      (pairs(0), pairs(3))
    }).groupByKey()

    val total = links.count()
    var ranks = links.mapValues(v => 10.0)

    for (i <- 1 to iter) {
      val ranksFromAirports = links.join(ranks).values.flatMap(toNodesPageRank => {
        val fromAirportRank = toNodesPageRank._2
        val outlinks = toNodesPageRank._1.size
        val toAirportslist = toNodesPageRank._1

        toAirportslist.map(topage => {
          val rankFromAirport = fromAirportRank / outlinks;
          (topage, rankFromAirport)
        })
      })

      ranks = ranksFromAirports.reduceByKey(_ + _).mapValues(rank => (1 - alpha) * rank + alpha / total)
    }
    ranks = ranks.sortBy(_._2, false)

    val result = ranks.map(airportRank => airportRank._1 + "\t" + airportRank._2)
    result.saveAsTextFile(outputDir)
    sc.stop()
  }
}

