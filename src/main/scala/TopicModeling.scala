import java.io.{File, PrintWriter}

import org.apache
import org.apache.spark
import org.apache.spark.{SparkContext, sql}
import org.apache.spark.ml.feature.{CountVectorizer, RegexTokenizer, StopWordsRemover, Tokenizer}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SQLContext._
import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.spark.ml.clustering.LDA


object TopicModeling {
  def main(args: Array[String]) {

    if (args.length != 2) {
      println("Usage: TopicModeling inputDir outputDir")
    }
    System.setProperty("hadoop.home.dir", args(2))//args(2))
    val inputPath = if (args.length > 0) args(0) else "G:\\IntelliJ\\PageRankFinal\\input\\74-0.txt"

    val outputDir = if (args.length > 1) args(1) else "G:\\IntelliJ\\PageRankFinal\\output1.txt"

//    val master = "local"
//    val sc = new SparkContext(master, "PageRank", System.getenv("SPARK_HOME"))
//    //    val sc = new SparkContext(new SparkConf().setAppName("Spark Count"))
//    sc.setLogLevel("ERROR")

    val book = new apache.spark.sql.SparkSession.Builder().getOrCreate().read.textFile(inputPath).toDF("text")
    val tkn = new Tokenizer().setInputCol("text").setOutputCol("textOut")
    val tokenized_df = tkn.transform(book)
    val remover = new StopWordsRemover()
      .setInputCol("textOut")
      .setOutputCol("filtered")


    val filtered_df = remover.transform(tokenized_df)
    val cv = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("features")
      .setVocabSize(10000)
      .setMinTF(0)
      .setMinDF(0)
      .setBinary(true)
    val cvFitted = cv.fit(filtered_df)
    val prepped = cvFitted.transform(filtered_df)

    val lda = new LDA().setK(5).setMaxIter(5)
//    println(lda.explainParams())
    val model = lda.fit(prepped)
    val vocabList = cvFitted.vocabulary
    val termsIdx2Str = udf { (termIndices: Seq[Int]) => termIndices.map(idx => vocabList(idx)) }
    val topics = model.describeTopics(maxTermsPerTopic = 5)
      .withColumn("terms", termsIdx2Str(col("termIndices")))
   // for (elem <- topics.select("topic", "terms", "termWeights").take(5)) {println(elem)}
//    val result = topics.select("topic", "terms", "termWeights")
//    result.rdd.map(_.toString()).saveAsTextFile(outputDir)

    val result1 = topics.select("topic", "terms", "termWeights")
//    result1.coalesce(1).write.mode(SaveMode.Overwrite).csv(outputDir)

    var sol = ""
    for (elem <- result1.take(5)) {var res = elem.toString()
              sol += res
      }
    val pw = new PrintWriter(new File(outputDir))
    pw.write(sol)
    pw.close()
//    result1.write.csv(outputDir)
//    sc.stop()
  }
}
