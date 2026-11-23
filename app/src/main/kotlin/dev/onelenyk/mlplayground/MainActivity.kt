package dev.onelenyk.mlplayground

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.CenterFocusStrong
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import dev.onelenyk.mlplayground.ui.benchmark.BenchmarkScreen
import dev.onelenyk.mlplayground.ui.benchmark.BenchmarkViewModel
import dev.onelenyk.mlplayground.ui.describe.DescribeScreen
import dev.onelenyk.mlplayground.ui.describe.DescribeViewModel
import dev.onelenyk.mlplayground.ui.scanner.ScannerScreen
import dev.onelenyk.mlplayground.ui.scanner.ScannerViewModel
import dev.onelenyk.mlplayground.ui.theme.MLPlaygroundTheme

private enum class AppTab(val title: String, val icon: ImageVector) {
  SCANNER("Сканер", Icons.Default.CenterFocusStrong),
  DESCRIBE("Опис AI", Icons.Default.AutoAwesome),
  BENCHMARK("Бенчмарк", Icons.Default.Speed),
}

class MainActivity : ComponentActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    enableEdgeToEdge()
    setContent {
      MLPlaygroundTheme {
        PlaygroundApp()
      }
    }
  }
}

@Composable
private fun PlaygroundApp() {
  val context = LocalContext.current
  var hasCameraPermission by remember {
    mutableStateOf(
      ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
        PackageManager.PERMISSION_GRANTED
    )
  }
  val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
    hasCameraPermission = granted
  }

  LaunchedEffect(Unit) {
    if (!hasCameraPermission) {
      launcher.launch(Manifest.permission.CAMERA)
    }
  }

  if (!hasCameraPermission) {
    PermissionGate(onRequest = { launcher.launch(Manifest.permission.CAMERA) })
    return
  }

  var currentTab by rememberSaveable { mutableStateOf(AppTab.SCANNER) }
  val scannerViewModel: ScannerViewModel = viewModel()
  val describeViewModel: DescribeViewModel = viewModel()
  val benchmarkViewModel: BenchmarkViewModel = viewModel()

  LaunchedEffect(currentTab) {
    scannerViewModel.setActive(currentTab == AppTab.SCANNER)
  }

  Scaffold(
    bottomBar = {
      NavigationBar {
        AppTab.entries.forEach { tab ->
          NavigationBarItem(
            selected = currentTab == tab,
            onClick = { currentTab = tab },
            icon = { Icon(tab.icon, contentDescription = tab.title) },
            label = { Text(tab.title) },
          )
        }
      }
    },
  ) { innerPadding ->
    val mod = Modifier.padding(innerPadding)
    when (currentTab) {
      AppTab.SCANNER -> ScannerScreen(viewModel = scannerViewModel, isActive = true, modifier = mod)
      AppTab.DESCRIBE -> DescribeScreen(viewModel = describeViewModel, isActive = true, modifier = mod)
      AppTab.BENCHMARK -> BenchmarkScreen(viewModel = benchmarkViewModel, isActive = true, modifier = mod)
    }
  }
}

@Composable
private fun PermissionGate(onRequest: () -> Unit) {
  Column(
    modifier = Modifier
      .fillMaxSize()
      .padding(24.dp),
    verticalArrangement = Arrangement.Center,
    horizontalAlignment = Alignment.CenterHorizontally,
  ) {
    Text(
      text = "Потрібен доступ до камери",
      style = MaterialTheme.typography.headlineSmall,
      textAlign = TextAlign.Center,
    )
    Spacer(Modifier.height(10.dp))
    Text(
      text = "ML Playground використовує камеру для розпізнавання предметів у реальному часі, порівняння рушіїв та опису сцени локальною нейромережею.",
      style = MaterialTheme.typography.bodyMedium,
      textAlign = TextAlign.Center,
    )
    Spacer(Modifier.height(20.dp))
    Button(onClick = onRequest) {
      Text("Надати доступ до камери")
    }
  }
}
