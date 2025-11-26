package com.example.llama

import android.app.ActivityManager
import android.app.DownloadManager
import android.content.ClipData
import android.content.ClipboardManager
import android.net.Uri
import android.os.Bundle
import android.os.StrictMode
import android.os.StrictMode.VmPolicy
import android.text.format.Formatter
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowUpward
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.foundation.layout.size
import androidx.compose.ui.graphics.Color
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.core.content.getSystemService
import com.example.llama.ui.theme.LlamaAndroidTheme
import java.io.File

class MainActivity(
    activityManager: ActivityManager? = null,
    downloadManager: DownloadManager? = null,
    clipboardManager: ClipboardManager? = null,
): ComponentActivity() {
    private val tag: String? = this::class.simpleName

    private val activityManager by lazy { activityManager ?: getSystemService<ActivityManager>()!! }
    private val downloadManager by lazy { downloadManager ?: getSystemService<DownloadManager>()!! }
    private val clipboardManager by lazy { clipboardManager ?: getSystemService<ClipboardManager>()!! }

    private val viewModel: MainViewModel by viewModels()

    // Get a MemoryInfo object for the device's current memory status.
    private fun availableMemory(): ActivityManager.MemoryInfo {
        return ActivityManager.MemoryInfo().also { memoryInfo ->
            activityManager.getMemoryInfo(memoryInfo)
        }
    }

    // 1. MainActivity类是程序打开之后运行的第一个类
    // 1. 并且打开之后对MainActivity实例化之后，执行OnCreate()方法
    override fun onCreate(savedInstanceState: Bundle?) {
        // 1.1 不需要关心
        super.onCreate(savedInstanceState)

        // 1.2 不需要关心
        StrictMode.setVmPolicy(
            VmPolicy.Builder(StrictMode.getVmPolicy())
                .detectLeakedClosableObjects()
                .build()
        )

        // 1.3 获取内存情况，并追加在viewModel的messages中
        val free = Formatter.formatFileSize(this, availableMemory().availMem)
        val total = Formatter.formatFileSize(this, availableMemory().totalMem)
        viewModel.log("Current memory: $free / $total")
        viewModel.log("Downloads directory: ${getExternalFilesDir(null)}")

        // 1.4 需要下载的模型列表
        val extFilesDir = getExternalFilesDir(null)
        val models = listOf(
            Downloadable(
                "TinyLlama 1.1B (int4, 669 MB)",
                Uri.parse("https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf?download=true"),  // 本地路径也可以写成 file://
                File(extFilesDir, "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")  // 保存/使用路径
            ),
            Downloadable(
                "Qwen2.5 0.5B (int8, 676 MB)",
                Uri.parse("https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf?download=true"),  // 本地路径也可以写成 file://
                File(extFilesDir, "qwen2.5-0.5b-instruct-q8_0.gguf")  // 保存/使用路径
            )
        )

        // 1.5 SetContent是一个函数
        // 1.5.1 程序入口是MainActivity，并且一进来实例化之后会马上执行OnCreate()
        // 1.5.1 一般MainActivity的OnCreate中都会有调用SetContent，因为SetContent之前的代码都是在创建或者声明变量，然后获取一些值，只有SetContent是真正在手机屏幕上展示内容
        // 1.5.2 func在声明的时候要求传入3个参数func(arg1, arg2, arg3){}，arg3是一个函数，然后在Kotlin的语法糖中，可以写成func(arg1, arg2){arg3}，原因是arg3的代码可能很长，这样会更好看
        // 1.5.2 如果func在声明的时候只要求传入1个参数，且这个参数是函数，就可以直接写成func{arg1}
        // 1.5.2 如果以上情况中所有参数都不是函数，只是一个变量（比如int，double或者某个类的实例），就不可以这样
        // 1.5.2 SetContent(arg1){}中，arg1就是一个函数，所以可以写成SetContent{arg1}

        // 1.5.3 SetContent(arg1:lambda){}，所以写成SetContent{arg1}
        // 1.5.3 LlamaAndroidTheme(arg1:lambda){}，所以写成LlamaAndroidTheme{arg1}
        // 1.5.3 Surface(arg1, arg2, arg3:lambda){}，所以写成Surface(arg1, arg2){arg3}
        setContent {
            LlamaAndroidTheme {
                // A surface container using the 'background' color from the theme
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    // 1.6 arg3:lambda
                    // 1.6.1 需要传入viewModel，剪切板，下载器，模型列表
                    // 1.6.2 arg3的具体声明代码在本类之后
                    MainCompose(
                        viewModel,
                        clipboardManager,
                        downloadManager,
                        models,
                    )
                }
            }
        }
    }
}

@Composable
fun MainCompose(
    viewModel: MainViewModel,
    clipboard: ClipboardManager,
    dm: DownloadManager,
    models: List<Downloadable>
) {
    // 1. Column(arg1, arg2, arg3:lambda){}
    // 1.1 可以写成Column(arg1, arg2){arg3}，只不过这个arg3其实可以写很多个按钮实例，最后被打包在一起作为一个arg3整体
    Column(
        horizontalAlignment = Alignment.CenterHorizontally, // 整体水平居中
        modifier = Modifier.fillMaxSize() // 占满整个屏幕
    ) {
        // // 1. 顶部AppBar
        // Row(
        //     verticalAlignment = Alignment.CenterVertically,
        //     modifier = Modifier
        //         .fillMaxWidth()
        //         .background(Color.White)
        //         .statusBarsPadding()
        //         .padding(vertical = 16.dp, horizontal = 16.dp)
        // ) {
        //     Box(
        //         modifier = Modifier
        //             .size(24.dp)
        //             .background(Color.Black, shape = RoundedCornerShape(12.dp))
        //     )
        //     Spacer(modifier = Modifier.width(8.dp))
        //     Text(
        //         text = "LlamaBot",
        //         color = Color.Black,
        //         style = MaterialTheme.typography.titleLarge
        //     )
        // }

        // 2. 对话内容
        val scrollState = rememberLazyListState()
        Box(modifier = Modifier.weight(1f).fillMaxWidth()) {
            // 1.1 LazyColumn是真正用于显示内容
            LazyColumn(
                state = scrollState,
                modifier = Modifier.fillMaxSize()
            ) {
                items(viewModel.messages.size) { index ->
                    val message = viewModel.messages[index]
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp, vertical = 4.dp),
                        horizontalArrangement = if (index % 2 == 0) Arrangement.End else Arrangement.Start
                    ) {
                        Text(
                            text = message,
                            style = MaterialTheme.typography.bodyLarge.copy(color = LocalContentColor.current),
                            modifier = Modifier
                                .background(MaterialTheme.colorScheme.primary.copy(alpha = 0.1f), shape = RoundedCornerShape(8.dp))
                                .padding(8.dp)
                        )
                    }
                }
            }

            // 1.2 LaunchedEffect用于计算出对话总长度来自动给滚动到底部
            LaunchedEffect(viewModel.messages) {
                snapshotFlow { viewModel.messages.lastOrNull() }
                    .collect { _ ->
                        if (viewModel.messages.isNotEmpty()) {
                            scrollState.animateScrollToItem(viewModel.messages.size - 1)
                        }
                    }
            }
        }

        // 3. 用户输入
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.fillMaxWidth(0.8f)
        ){
            // 3.1 获取用户的输入，并且赋值给viewModel的message属性
            OutlinedTextField(
                value = viewModel.message,
                onValueChange = { viewModel.updateMessage(it) },
                label = { Text("Message") },
                modifier = Modifier.fillMaxWidth(0.8f), // 可选，限制宽度并居中
                shape = RoundedCornerShape(24.dp) // 设置圆角
            )
            Spacer(modifier = Modifier.width(8.dp))

            IconButton(
                onClick = { viewModel.send() },
                modifier = Modifier
                    .background(
                        color = MaterialTheme.colorScheme.primary,
                        shape = RoundedCornerShape(50)
                    )
                    .size(48.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.ArrowUpward,
                    contentDescription = "Send",
                    tint = MaterialTheme.colorScheme.onPrimary
                )
            }
        }

        // 3. 下载按钮
        Column(
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            for (model in models) {
                Downloadable.Button(viewModel, dm, model)
            }
        }
    }
}
